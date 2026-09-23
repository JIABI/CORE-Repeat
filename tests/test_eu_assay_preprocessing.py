import numpy as np
import pytest

from opal2.eu_assay_preprocessing import fit_control_space, apply_control_space, measurement_columns


def source():
    rng = np.random.default_rng(41)
    values = rng.normal(size=(22, 12))
    names = [f'Cells_Intensity_Feature_{i}' for i in range(10)]
    names += ['Metadata_Object_Count', 'Cells_Parent_Nuclei']
    plates = np.array(['P1']*11+['P2']*11)
    dmso = np.array(([True]*8+[False]*3)*2)
    return values, plates, dmso, names


def test_assay_space_uses_only_controls_and_not_drug_outcomes():
    values, plates, dmso, names = source()
    before = fit_control_space(values, plates, dmso, names)
    altered = values.copy()
    altered[~dmso] *= 10000
    after = fit_control_space(altered, plates, dmso, names)
    assert before == after
    assert len(before['feature_names']) == 10
    actual, audit = apply_control_space(values, plates, before, required_wells=~dmso)
    assert actual.shape == (22, 10)
    assert np.max(np.abs(actual)) <= 10
    assert not audit['objects_removed']


def test_degenerate_control_coordinate_removed_and_bad_drug_not_deleted():
    values, plates, dmso, names = source()
    values[(plates == 'P1') & dmso, 0] = 5
    space = fit_control_space(values, plates, dmso, names)
    assert 0 not in space['selected_indices']
    values[9, 1:4] = np.nan
    with pytest.raises(ValueError, match='invalid drug wells'):
        apply_control_space(values, plates, space, required_wells=~dmso)


def test_eu_abbreviated_compartments_not_silently_discarded():
    names=['Cells_AreaShape_Area','Cyto_Intensity_MeanIntensity','Nuc_Texture_Variance',
           'Cytoplasm_AreaShape_Area','Nuclei_AreaShape_Area','Nuc_Number_Object_Number',
           'Cyto_Parent_Nuclei','Metadata_Object_Count']
    assert measurement_columns(names).tolist()==[True,True,True,True,True,False,False,False]
