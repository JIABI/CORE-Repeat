import pytest
from opal2.eu_fit_dataset import validate_export_metadata, chemistry_for, build_dataset


def test_export_identity_guard():
    allow = [dict(well_id='p:A01', plate_uid='p', object_id='fit1', resource_kind='FIT_COMPOUND', measurement_role='X')]
    metadata = [dict(allow[0], export_row_index='0')]
    assert list(validate_export_metadata(metadata, allow)) == ['p:A01']
    bad = [dict(metadata[0], object_id='reserved')]
    with pytest.raises(ValueError): validate_export_metadata(bad, allow)
    with pytest.raises(ValueError): validate_export_metadata(metadata+metadata, allow)


def test_morgan_schema(tmp_path):
    path = tmp_path/'identity.csv'
    path.write_text('object_id,smiles\na,CCO\nb,CCN\n')
    fp, mask = chemistry_for(['a','b'], path)
    assert fp.shape == (2,513) and mask.all()
    assert (fp[:,512] == 1).all() and (fp[:,:512].sum(1)>0).all()


def test_incomplete_inventory_blocks_before_measurement_open(tmp_path):
    ingest=tmp_path/'ingest';ingest.mkdir()
    (ingest/'audit.json').write_text('{"dataset_incomplete_training_blocked":true,"missing_allowed_wells":13}')
    with pytest.raises(ValueError,match='13 planned wells absent'):
        build_dataset(tmp_path/'no_plan',ingest,tmp_path/'no_chemistry',tmp_path/'output')
    assert not (tmp_path/'output').exists()
