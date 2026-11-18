import copy

import pytest

from lightagent.demo import inventory
from lightagent.light_rules import effective_energy, illumination_group_metadata
from lightagent.prompts import INVENTORY, BUILD, REVIEW
from lightagent.state import initialize, apply_patch, validate


def daylight_inventory():
    inv = inventory()
    sun, sky = inv['lights']
    sun.update(source_type='direct_sun', illumination_group_id='daylight')
    sun['elements'] = [{'id': 'solar_disk', 'kind': 'SUN', 'direction': [.2, 1., -1.],
                        'energy': 2., 'color': [1., .95, .9]}]
    sky.update(source_type='sky_diffuse', illumination_group_id='daylight', approximation={
        'kind': 'sky_diffuse_area', 'via_objects': [],
        'note': 'Diffuse sky approximation; no specific opening identified in synthetic fixture'})
    inv['illumination_groups'] = [{'id': 'daylight', 'type': 'daylight', 'label': 'Outdoor daylight',
                                  'sun_light_id': sun['id'], 'gain': 1., 'enabled': True}]
    return inv


def edit(state, **fields):
    return apply_patch(state, {'base_revision': state['revision'], 'reason': 'daylight editing test', **fields}, 0)


def energies(state):
    return {lid: effective_energy(state, light, light['elements'][0]) for lid, light in state['lights'].items()}


def test_group_gain_scales_energy_without_changing_base_parameters():
    state = initialize(daylight_inventory())
    changed = edit(state, upsert_illumination_groups=[{'id': 'daylight', 'gain': .5}])
    assert energies(changed) == {lid: value*.5 for lid, value in energies(state).items()}
    assert changed['lights'] == state['lights']
    assert changed['world'] == state['world']
    again = edit(changed, upsert_illumination_groups=[{'id': 'daylight', 'gain': .25}])
    assert energies(again) == {lid: value*.25 for lid, value in energies(state).items()}
    off = edit(state, upsert_illumination_groups=[{'id': 'daylight', 'enabled': False}])
    assert set(energies(off).values()) == {0.}


def test_direct_sun_edits_do_not_move_or_switch_off_sky_proxy():
    state = initialize(daylight_inventory())
    elem = copy.deepcopy(state['lights']['light_key']['elements'][0])
    elem['direction'] = [-.3, 1., -.8]
    changed = edit(state, upsert_lights=[{'id': 'light_key', 'elements': [elem]}])
    assert changed['lights']['light_fill'] == state['lights']['light_fill']
    off = edit(state, upsert_lights=[{'id': 'light_key', 'status': 'inactive'}])
    assert energies(off)['light_key'] == 0.
    assert energies(off)['light_fill'] == energies(state)['light_fill'] > 0.


@pytest.mark.parametrize('changes', [
    {'sun_light_id': 'missing'}, {'sun_light_id': 'light_fill'},
    {'gain': -.1}, {'gain': float('nan')}, {'gain': float('inf')}, {'enabled': 'false'},
])
def test_invalid_group_edits_are_transactional(changes):
    state = initialize(daylight_inventory())
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        edit(state, upsert_illumination_groups=[{'id': 'daylight', **changes}])
    assert state == before


def test_proxy_requires_correct_source_and_opening_references():
    inv = daylight_inventory()
    inv['lights'][1].pop('approximation')
    with pytest.raises(ValueError, match='requires illumination_group_id'):
        initialize(inv)
    state = initialize(daylight_inventory())
    with pytest.raises(ValueError, match='only contain'):
        edit(state, upsert_lights=[{'id': 'light_fill', 'source_type': 'artificial'}])
    invalid = copy.deepcopy(state['lights']['light_fill']['approximation'])
    invalid['via_objects'] = ['missing_window']
    with pytest.raises(ValueError, match='existing object IDs'):
        edit(state, upsert_lights=[{'id': 'light_fill', 'approximation': invalid}])


def test_sky_only_group_does_not_require_invented_sun():
    inv = daylight_inventory()
    inv['lights'] = inv['lights'][1:]
    inv['illumination_groups'][0]['sun_light_id'] = None
    state = initialize(inv)
    assert energies(state)['light_fill'] > 0.
    metadata = illumination_group_metadata(state)['daylight']
    assert metadata['members'] == ['light_fill']
    assert metadata['sun_light_id'] is None
    assert metadata['world_included'] is False


def test_link_existing_lights_and_create_group_in_one_patch():
    state = initialize(inventory())
    inv = daylight_inventory()
    changed = edit(state, upsert_lights=inv['lights'], upsert_illumination_groups=inv['illumination_groups'])
    assert changed['lights']['light_key']['illumination_group_id'] == 'daylight'
    assert set(illumination_group_metadata(changed)['daylight']['members']) == {'light_key', 'light_fill'}


def test_historical_state_readable_and_edited_proxy_must_gain_link():
    state = initialize(inventory())
    state.pop('illumination_groups')
    state['lights']['light_fill']['source_type'] = 'sky_diffuse'
    validate(state)
    with pytest.raises(ValueError, match='requires illumination_group_id'):
        edit(state, upsert_lights=[{'id': 'light_fill', 'notes': 'edit legacy proxy'}])


def test_all_roles_receive_group_editing_and_semantics_contract():
    for prompt in (INVENTORY, BUILD, REVIEW):
        assert 'DAYLIGHT EDITING ASSOCIATION' in prompt
        assert 'upsert_illumination_groups' in prompt
        assert 'World/HDRI is NOT included' in prompt
