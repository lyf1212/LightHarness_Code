import json

import pytest

from lightagent.llm import Client, sse_events


class Response:
    ok = True
    status_code = 200

    def __init__(self, events=(), tail_error=False, body=None):
        self.events = events
        self.tail_error = tail_error
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self, chunk_size):
        assert chunk_size == 1
        yield b': heartbeat'
        yield b''
        for event in self.events:
            yield b'data: ' + json.dumps(event).encode()
            yield b''
        if self.tail_error:
            raise AssertionError('Must stop at response.completed without waiting for EOF')

    def json(self):
        return self.body


def setup_api(monkeypatch, response):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://example.invalid/v1')

    def post(url, **kwargs):
        assert url == 'https://example.invalid/v1/responses'
        assert kwargs['json']['reasoning'] == {'effort': 'xhigh'}
        assert kwargs['json']['store'] is False
        assert kwargs['json']['model'] == 'gpt-6-astra'
        return response

    monkeypatch.setattr('requests.post', post)
    return Client({'api': 'responses', 'model': 'gpt-6-astra', 'reasoning_effort': 'xhigh', 'max_retries': 0})


def completed(text='{"ok":true}'):
    return {'status': 'completed', 'output': [{'content': [{'type': 'output_text', 'text': text}]}],
            'usage': {'output_tokens': 10}}


def test_stream_writes_deltas_and_stops_at_terminal(tmp_path, monkeypatch):
    events = [{'type': 'response.created'},
              {'type': 'response.output_text.delta', 'delta': '{"ok":'},
              {'type': 'response.output_text.delta', 'delta': 'true}'},
              {'type': 'response.completed', 'response': completed()}]
    client = setup_api(monkeypatch, Response(events, tail_error=True))
    folder = tmp_path / 'call'
    assert client.ask(folder, 'system', {}) == {'ok': True}
    assert (folder / 'output.partial.txt').read_text() == '{"ok":true}'
    timing = json.loads((folder / 'timing.json').read_text())
    assert timing['event_count'] == 4
    assert 0 <= timing['first_event_seconds'] <= timing['first_text_seconds'] <= timing['seconds']
    assert timing['status'] == 'completed'
    assert 'secret-test-key' not in ''.join(p.read_text() for p in folder.glob('*') if p.is_file())


@pytest.mark.parametrize('kind', ['response.failed', 'response.incomplete', 'error', 'eof'])
def test_partial_failure_is_archived_and_rejected(tmp_path, monkeypatch, kind):
    events = [{'type': 'response.output_text.delta', 'delta': '{"ok":'}]
    if kind != 'eof':
        events.append({'type': kind, 'response': {'status': 'failed'}})
    client = setup_api(monkeypatch, Response(events))
    folder = tmp_path / 'call'
    with pytest.raises(RuntimeError):
        client.ask(folder, 'system', {})
    assert (folder / 'output.partial.txt').read_text() == '{"ok":'
    assert not (folder / 'parsed.json').exists()
    assert json.loads((folder / 'timing.json').read_text())['status'] == 'failed'


def test_completed_without_deltas_uses_final_output(tmp_path, monkeypatch):
    client = setup_api(monkeypatch, Response([{'type': 'response.completed', 'response': completed()}]))
    assert client.ask(tmp_path / 'call', 'system', {}) == {'ok': True}


def test_nonstream_baseline(tmp_path, monkeypatch):
    client = setup_api(monkeypatch, Response(body=completed()))
    client.config['stream'] = False
    assert client.ask(tmp_path / 'call', 'system', {}) == {'ok': True}
    timing = json.loads((tmp_path / 'call/timing.json').read_text())
    assert timing['first_text_seconds'] is not None
    assert timing['event_count'] == 0


def test_multiline_sse_and_eof():
    class Frames:
        def iter_lines(self, chunk_size):
            return iter([b': ping', b'event: response.output_text.delta', b'data: {"type":',
                         b'data: "response.output_text.delta", "delta":"x"}', b'',
                         b'data: {"type":"response.completed"}'])
    assert list(sse_events(Frames())) == [{'type': 'response.output_text.delta', 'delta': 'x'},
                                        {'type': 'response.completed'}]


def test_done_marker():
    class Frames:
        def iter_lines(self, chunk_size):
            yield b'data: [DONE]'
            yield b''
            raise AssertionError('Read past DONE')
    assert list(sse_events(Frames())) == []


@pytest.mark.parametrize('stream', [True, False])
def test_chat_transport_still_collects_json(tmp_path, monkeypatch, stream):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://example.invalid/v1')
    if stream:
        response = Response([{'choices': [{'delta': {'content': '{"ok":true}'}, 'finish_reason': None}]},
                             {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}])
    else:
        response = Response(body={'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}]})

    def post(url, **kwargs):
        assert url.endswith('/chat/completions')
        assert kwargs['json']['reasoning_effort'] == 'xhigh'
        return response

    monkeypatch.setattr('requests.post', post)
    client = Client({'api': 'chat', 'model': 'test', 'stream': stream, 'reasoning_effort': 'xhigh'})
    assert client.ask(tmp_path / 'call', 'system', {}) == {'ok': True}


def test_reconnect_preserves_failed_stream_and_discards_partial_text(tmp_path, monkeypatch):
    import requests
    class Interrupted(Response):
        def iter_lines(self, chunk_size):
            yield b'data: {"type":"response.output_text.delta","delta":"BAD PARTIAL"}'
            yield b''
            raise requests.exceptions.ChunkedEncodingError('link lost secret-test-key')
    client = setup_api(monkeypatch, Response())
    client.config.update(max_retries=2, retry_initial_delay=0)
    replies = iter([Interrupted(), Response([{'type': 'response.completed', 'response': completed()}])])
    monkeypatch.setattr('requests.post', lambda *a, **kw: next(replies))
    folder = tmp_path / 'call'
    assert client.ask(folder, 'system', {}) == {'ok': True}
    assert (folder / 'network/attempt_000/output.partial.txt').read_text() == 'BAD PARTIAL'
    assert (folder / 'output.txt').read_text() == '{"ok":true}'
    assert json.loads((folder / 'timing.json').read_text())['network_attempts'] == 2
    assert 'secret-test-key' not in ''.join(p.read_text() for p in folder.rglob('*') if p.is_file())


@pytest.mark.parametrize('code, calls', [(429, 2), (503, 2), (401, 1), (403, 1), (400, 1)])
def test_http_retry_classification_and_retry_after(tmp_path, monkeypatch, code, calls):
    class Bad(Response):
        ok = False
        status_code = code
        text = 'error secret-test-key'
        headers = {'Retry-After': '3'}
    client = setup_api(monkeypatch, Response())
    client.config.update(stream=False, max_retries=1, retry_initial_delay=0)
    seen, sleeps = [], []
    def post(*args, **kwargs):
        seen.append(1)
        return Bad() if len(seen) == 1 else Response(body=completed())
    monkeypatch.setattr('requests.post', post)
    monkeypatch.setattr('lightagent.llm.time.sleep', sleeps.append)
    if calls == 2:
        assert client.ask(tmp_path / 'call', 'system', {}) == {'ok': True}
        assert sleeps == [3]
    else:
        with pytest.raises(RuntimeError, match=f'HTTP {code}'):
            client.ask(tmp_path / 'call', 'system', {})
        assert not sleeps
    assert len(seen) == calls


def test_nonstream_insufficient_balance_is_visible_redacted_and_not_retried(tmp_path, monkeypatch):
    class InsufficientBalance(Response):
        ok = False
        status_code = 403
        text = json.dumps({'error': {'message': 'Your account balance is insufficient. secret-test-key',
                                    'type': 'Aihubmix_api_error'}})
    client = setup_api(monkeypatch, InsufficientBalance())
    client.config.update(stream=False, max_retries=20)
    sleeps = []
    monkeypatch.setattr('lightagent.llm.time.sleep', sleeps.append)
    folder = tmp_path / 'call'
    with pytest.raises(RuntimeError, match='HTTP 403: Your account balance is insufficient') as error:
        client.ask(folder, 'system', {})
    assert 'secret-test-key' not in str(error.value)
    assert not sleeps
    timing = json.loads((folder / 'timing.json').read_text())
    assert timing['stream'] is False
    assert timing['network_attempts'] == 1
    assert 'balance is insufficient' in timing['error']
    assert not (folder / 'parsed.json').exists()
    assert 'secret-test-key' not in ''.join(p.read_text() for p in folder.rglob('*') if p.is_file())


def test_proxy_retry_logs_cause_and_clears_error_after_recovery(tmp_path, monkeypatch, capsys):
    import requests
    client = setup_api(monkeypatch, Response())
    client.config.update(stream=False, max_retries=1, retry_initial_delay=0)
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ProxyError('Cannot connect to proxy secret-test-key')
        return Response(body=completed())
    monkeypatch.setattr('requests.post', post)
    folder = tmp_path / 'call'
    assert client.ask(folder, 'system', {}) == {'ok': True}
    output = capsys.readouterr().out
    assert 'ProxyError: Cannot connect to proxy' in output
    assert 'secret-test-key' not in output
    timing = json.loads((folder / 'timing.json').read_text())
    assert timing['network_attempts'] == 2
    assert 'error' not in timing and 'error_type' not in timing


def test_network_retry_exhaustion_is_bounded(tmp_path, monkeypatch):
    import requests
    client = setup_api(monkeypatch, Response())
    client.config.update(max_retries=2, retry_initial_delay=0)
    calls = []
    def offline(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError('offline')
    monkeypatch.setattr('requests.post', offline)
    with pytest.raises(RuntimeError, match='reconnect exhausted after 3 attempts'):
        client.ask(tmp_path / 'call', 'system', {})
    assert len(calls) == 3
    assert not (tmp_path / 'call/parsed.json').exists()


@pytest.mark.parametrize('kind', ['eof', 'upstream_error'])
def test_stream_transient_error_reconnects(tmp_path, monkeypatch, kind):
    bad = [] if kind == 'eof' else [{'type': 'error', 'code': 'upstream_error'}]
    client = setup_api(monkeypatch, Response())
    client.config.update(max_retries=1, retry_initial_delay=0)
    replies = iter([Response(bad), Response([{'type': 'response.completed', 'response': completed()}])])
    monkeypatch.setattr('requests.post', lambda *a, **kw: next(replies))
    assert client.ask(tmp_path / 'call', 'system', {}) == {'ok': True}
