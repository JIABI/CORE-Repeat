import numpy as np
from opal2.lkcp_module_switch_experiment import grouped_roles


def test_condition_roles_hold_complete_chemical_groups():
    source=np.repeat([f'g{i:03}' for i in range(50)],3)
    groups=np.r_[source,source]
    roles=grouped_roles(groups,len(source))
    seen=np.zeros(len(groups),int)
    for row in roles:
        parts=[set(groups[row[k]]) for k in ('fit','covfit','calibration','query')]
        assert all(not parts[i]&parts[j] for i in range(4) for j in range(i+1,4))
        assert all(np.all(row[k]<len(source)) for k in ('fit','covfit','calibration'))
        assert np.all(row['query']>=len(source))
        seen[row['query']]+=1
    assert not seen[:len(source)].any()
    assert np.all(seen[len(source):]==1)
