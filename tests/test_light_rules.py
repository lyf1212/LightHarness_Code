import copy
import math
import pytest

from lightagent.demo import inventory
from lightagent.state import initialize, apply_patch, validate
from lightagent.light_rules import validate_light_sources
from lightagent.prompts import INVENTORY, BUILD, REVIEW


def sun_inventory():
    inv = inventory()
    sun = copy.deepcopy(inv['lights'][0])
    sun.update(id='sun', label='direct sunlight through the windows', source_type='direct_sun',
               environment_relation='independent')
    sun['elements'] = [{'id': 'solar_disk', 'kind': 'SUN', 'direction': [.2, 1, -1],
                        'energy': 2., 'color': [1., .95, .9], 'angle': .00918, 'enabled': True}]
    inv['lights'] = [sun]
    inv['world'] = {'mode': 'evidence_only', 'strength': 0., 'rotation_z': 0.}
    return inv


@pytest.mark.parametrize('kind', ['AREA', 'POINT', 'SPOT', 'EMISSIVE_MESH', 'GEOMETRY'])
def test_direct_sun_cannot_use_substitute_emitters(kind):
    state = initialize(sun_inventory())
    state['lights']['sun']['elements'][0]['kind'] = kind
    with pytest.raises(ValueError, match='exactly one SUN'):
        validate_light_sources(state)


def test_initialization_requires_physical_source_classification():
    inv = sun_inventory()
    inv['lights'][0].pop('source_type')
    with pytest.raises(ValueError, match='source_type'):
        initialize(inv)


def test_invalid_update_is_transactional_and_cannot_relabel_sun():
    state = initialize(sun_inventory())
    before = copy.deepcopy(state)
    area = inventory()['lights'][0]['elements']
    with pytest.raises(ValueError, match='exactly one SUN'):
        apply_patch(state, {'base_revision': 0, 'reason': 'brighten window',
                           'upsert_lights': [{'id': 'sun', 'elements': area}]}, 0)
    assert state == before
    with pytest.raises(ValueError, match='cannot relabel'):
        apply_patch(state, {'base_revision': 0, 'reason': 'rename',
                           'upsert_lights': [{'id': 'sun', 'source_type': 'artificial', 'elements': area}]}, 0)
    assert state == before
    retired = apply_patch(state, {'base_revision': 0, 'reason': 'no direct sun evidence',
                                 'retire_lights': ['sun']}, 0)
    assert retired['lights']['sun']['status'] == 'retired'


def test_direction_intensity_angle_updates_need_no_position():
    state = initialize(sun_inventory())
    elem = copy.deepcopy(state['lights']['sun']['elements'][0])
    elem.update(direction=[-.4, 1, -.8], energy=3., angle=.01)
    changed = apply_patch(state, {'base_revision': 0, 'reason': 'match cast shadows',
                                 'upsert_lights': [{'id': 'sun', 'elements': [elem]}]}, 0)
    assert changed['lights']['sun']['elements'][0] == elem
    assert changed['revision'] == 1


def test_sun_rejects_area_parameters_and_duplicate_active_sources():
    inv = sun_inventory()
    inv['lights'][0]['elements'][0]['size'] = [3, 3]
    with pytest.raises(ValueError, match='not'):
        initialize(inv)
    inv = sun_inventory()
    second = copy.deepcopy(inv['lights'][0]); second['id'] = 'sun_other_window'
    inv['lights'].append(second)
    with pytest.raises(ValueError, match='one emitting SUN'):
        initialize(inv)
    inv['lights'][1]['status'] = 'hypothesis'
    initialize(inv)  # Alternative hypotheses do not emit.


def test_sky_and_artificial_panels_remain_valid_but_not_solar_duplicates():
    inv = sun_inventory()
    fill = copy.deepcopy(inventory()['lights'][0]); fill['source_type'] = 'sky_diffuse'
    inv['illumination_groups'] = [{'id': 'daylight', 'type': 'daylight', 'label': 'daylight',
                                  'sun_light_id': 'sun', 'gain': 1., 'enabled': True}]
    inv['lights'][0]['illumination_group_id'] = 'daylight'
    fill.update(illumination_group_id='daylight', approximation={
        'kind': 'sky_diffuse_area', 'via_objects': [], 'note': 'Diffuse sky; opening uncertain'})
    inv['lights'].append(fill)
    state = initialize(inv)
    validate(state)
    inv['lights'][0]['environment_relation'] = 'represented_in_hdri'
    inv['world'].update(mode='world', strength=1.)
    with pytest.raises(ValueError, match='duplicates'):
        initialize(inv)


def test_legacy_records_readable_but_must_be_classified_when_edited():
    state = initialize(inventory())
    state['lights']['light_key'].pop('source_type')
    validate(state)
    patch = {'base_revision': 0, 'reason': 'update',
             'upsert_lights': [{'id': 'light_key', 'notes': 'change'}]}
    with pytest.raises(ValueError, match='legacy'):
        apply_patch(state, patch, 0)
    patch['upsert_lights'][0]['source_type'] = 'artificial'
    apply_patch(state, patch, 0)


@pytest.mark.parametrize('angle', [float('nan'), float('inf'), -.01, math.pi+.01])
def test_nonphysical_angle_domain_rejected(angle):
    inv = sun_inventory(); inv['lights'][0]['elements'][0]['angle'] = angle
    with pytest.raises(ValueError):
        initialize(inv)


def test_all_three_model_roles_receive_sun_rule():
    for prompt in [INVENTORY, BUILD, REVIEW]:
        assert 'PHYSICAL SUN RULE' in prompt
        assert 'Direct solar illumination MUST' in prompt
