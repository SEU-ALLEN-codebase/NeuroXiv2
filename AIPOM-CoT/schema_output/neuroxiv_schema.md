# NeuroXiv 2.0 知识图谱 Schema

## 统计概览

- **总节点数**: 34,771
- **总关系数**: 258,359,096
- **节点类型数**: 8
- **关系类型数**: 12

## 节点类型

### Class

**数量**: 28

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `dominant_neurotransmitter_type` | str | 28/28 (100.0%) |  |
| `id` | int | 28/28 (100.0%) | 301 |
| `markers` | str | 28/28 (100.0%) |  |
| `name` | str | 28/28 (100.0%) | 01 IT-ET Glut |
| `neighborhood` | str | 28/28 (100.0%) | Pallium-Glut |
| `number_of_child_types` | int | 28/28 (100.0%) | 26 |
| `number_of_neurons` | int | 28/28 (100.0%) | 657089 |
| `tran_id` | int | 28/28 (100.0%) | 1 |

### Cluster

**数量**: 5,204

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `anatomical_annotation` | str | 5204/5204 (100.0%) | CLA-EPd-iCTX |
| `broad_region_distribution` | str | 5204/5204 (100.0%) | Isocortex:0.57,CTXsp:0.21,OLF:... |
| `dominant_neurotransmitter_type` | str | 5204/5204 (100.0%) | Glut |
| `id` | int | 5204/5204 (100.0%) | 1928 |
| `markers` | str | 5204/5204 (100.0%) | Car3,Satb2,Lgr5 |
| `name` | str | 5204/5204 (100.0%) | 0001 CLA-EPd-CTX Car3 Glut_1 |
| `neuropeptide_mark_genes` | str | 5204/5204 (100.0%) | Cck:9.5,Cartpt:3.7 |
| `neurotransmitter_mark_genes` | str | 5204/5204 (100.0%) | Slc17a7:9.91,Slc17a6:4.87 |
| `number_of_neurons` | int | 5204/5204 (100.0%) | 3450 |
| `tran_id` | int | 5204/5204 (100.0%) | 1 |
| `transcription_factor_markers` | str | 5204/5204 (100.0%) | Satb2,Nr2f2,Rarb,Rorb,Pou6f2,N... |
| `within_subclass_markers` | str | 5204/5204 (100.0%) | Lgr5 |

### ME_Subregion

**数量**: 628

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `acronym` | str | 628/628 (100.0%) | MOp2/3-ME1 |
| `me_subregion_id` | str | 628/628 (100.0%) | 943_953 |
| `name` | str | 628/628 (100.0%) | Primary motor area, Layer 2/3,... |
| `parent_subregion` | str | 628/628 (100.0%) | MOp2/3 |
| `rgb_triplet` | list | 628/628 (100.0%) | [31, 157, 90] |

### Neuron

**数量**: 26,740

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `axonal_25pct_euclidean_distance` | float | 26740/26740 (100.0%) | 584.3193034998797 |
| `axonal_25pct_path_distance` | float | 26740/26740 (100.0%) | 1105.2999347919235 |
| `axonal_2d_density` | float | 26740/26740 (100.0%) | 0.0029514494745135 |
| `axonal_3d_density` | float | 26740/26740 (100.0%) | 3.72520283762836e-06 |
| `axonal_50pct_euclidean_distance` | float | 26740/26740 (100.0%) | 2347.4536020470114 |
| `axonal_50pct_path_distance` | float | 26740/26740 (100.0%) | 3280.4706589956795 |
| `axonal_75pct_euclidean_distance` | float | 26740/26740 (100.0%) | 4470.67563409731 |
| `axonal_75pct_path_distance` | float | 26740/26740 (100.0%) | 6106.533093560937 |
| `axonal_area` | float | 26740/26740 (100.0%) | 17301329.54704132 |
| `axonal_average_bifurcation_angle_local` | float | 26740/26740 (100.0%) | 63.01808678438939 |
| `axonal_average_bifurcation_angle_remote` | float | 26740/26740 (100.0%) | 53.80583983694522 |
| `axonal_average_contraction` | float | 26740/26740 (100.0%) | 0.8506856780636308 |
| `axonal_average_euclidean_distance` | float | 26740/26740 (100.0%) | 2432.846293579268 |
| `axonal_average_path_distance` | float | 26740/26740 (100.0%) | 3605.5456889227694 |
| `axonal_center_shift` | float | 26740/26740 (100.0%) | 1542.2061694941972 |
| `axonal_depth` | float | 26740/26740 (100.0%) | 1279.2896118164062 |
| `axonal_depth_95ci` | float | 26740/26740 (100.0%) | 949.314730834961 |
| `axonal_flatness` | float | 26740/26740 (100.0%) | 5.593351030952896 |
| `axonal_flatness_95ci` | float | 26740/26740 (100.0%) | 7.089090255344583 |
| `axonal_height` | float | 26740/26740 (100.0%) | 7155.515869140625 |
| `axonal_height_95ci` | float | 26740/26740 (100.0%) | 6729.777807617187 |
| `axonal_max_branch_order` | float | 26740/26740 (100.0%) | 16.0 |
| `axonal_max_euclidean_distance` | float | 26740/26740 (100.0%) | 5747.922962338587 |
| `axonal_max_path_distance` | float | 26740/26740 (100.0%) | 9451.25458508074 |
| `axonal_number_of_bifurcations` | float | 26740/26740 (100.0%) | 70.0 |
| `axonal_relative_center_shift` | float | 26740/26740 (100.0%) | 0.2683066874067391 |
| `axonal_slimness` | float | 26740/26740 (100.0%) | 0.5293459417046502 |
| `axonal_slimness_95ci` | float | 26740/26740 (100.0%) | 0.5190823297809602 |
| `axonal_total_length` | float | 26740/26740 (100.0%) | 47400.77481879247 |
| `axonal_volume` | float | 26740/26740 (100.0%) | 15023343006.908575 |
| `axonal_width` | float | 26740/26740 (100.0%) | 3787.743286132813 |
| `axonal_width_95ci` | float | 26740/26740 (100.0%) | 3493.3087432861325 |
| `base_region` | str | 26740/26740 (100.0%) | CA |
| `celltype` | str | 26740/26740 (100.0%) | CA1 |
| `dendritic_25pct_euclidean_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_25pct_path_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_2d_density` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_3d_density` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_50pct_euclidean_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_50pct_path_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_75pct_euclidean_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_75pct_path_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_area` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_average_bifurcation_angle_local` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_average_bifurcation_angle_remote` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_average_contraction` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_average_euclidean_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_average_path_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_center_shift` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_depth` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_depth_95ci` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_flatness` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_flatness_95ci` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_height` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_height_95ci` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_max_branch_order` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_max_euclidean_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_max_path_distance` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_number_of_bifurcations` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_relative_center_shift` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_slimness` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_slimness_95ci` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_total_length` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_volume` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_width` | float | 26740/26740 (100.0%) | 0.0 |
| `dendritic_width_95ci` | float | 26740/26740 (100.0%) | 0.0 |
| `name` | str | 26740/26740 (100.0%) | ION_full_201770_008_CCFv3 |
| `neuron_id` | str | 26740/26740 (100.0%) | ION_full_201770_008_CCFv3 |

### Region

**数量**: 337

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `acronym` | str | 337/337 (100.0%) | Region_0 |
| `axonal_bifurcation_remote_angle` | float | 337/337 (100.0%) | 0.0 |
| `axonal_branches` | float | 337/337 (100.0%) | 0.0 |
| `axonal_length` | float | 337/337 (100.0%) | 0.0 |
| `axonal_maximum_branch_order` | float | 337/337 (100.0%) | 0.0 |
| `dendritic_bifurcation_remote_angle` | float | 337/337 (100.0%) | 0.0 |
| `dendritic_branches` | float | 337/337 (100.0%) | 0.0 |
| `dendritic_length` | float | 337/337 (100.0%) | 0.0 |
| `dendritic_maximum_branch_order` | float | 337/337 (100.0%) | 0.0 |
| `full_name` | str | 337/337 (100.0%) | Region_0 |
| `name` | str | 337/337 (100.0%) | Region_0 |
| `number_of_apical_dendritic_morphologies` | int | 337/337 (100.0%) | 0 |
| `number_of_axonal_morphologies` | int | 337/337 (100.0%) | 0 |
| `number_of_dendritic_morphologies` | int | 337/337 (100.0%) | 0 |
| `number_of_neuron_morphologies` | int | 337/337 (100.0%) | 0 |
| `number_of_transcriptomic_neurons` | int | 337/337 (100.0%) | 104847 |
| `parent_id` | int | 337/337 (100.0%) | 0 |
| `region_id` | int | 337/337 (100.0%) | 0 |

### Subclass

**数量**: 314

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `dominant_neurotransmitter_type` | str | 314/314 (100.0%) | Glut |
| `id` | int | 314/314 (100.0%) | 351 |
| `markers` | str | 314/314 (100.0%) | Car3,Slc17a7 |
| `name` | str | 314/314 (100.0%) | 001 CLA-EPd-CTX Car3 Glut |
| `neighborhood` | str | 314/314 (100.0%) | Pallium-Glut |
| `number_of_child_types` | int | 314/314 (100.0%) | 2 |
| `number_of_neurons` | int | 314/314 (100.0%) | 18996 |
| `tran_id` | int | 314/314 (100.0%) | 1 |
| `transcription_factor_markers` | str | 314/314 (100.0%) | Cux2,Satb2,Nr4a2,Zfhx4,Pou6f2 |
| `viz_layer` | int | 21/314 (6.7%) | 2 |
| `viz_x` | float | 21/314 (6.7%) | -11.996800255990312 |
| `viz_y` | float | 21/314 (6.7%) | -299.76003199829336 |

### Subregion

**数量**: 365

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `acronym` | str | 365/365 (100.0%) | FRP1 |
| `has_me_children` | bool | 365/365 (100.0%) | False |
| `name` | str | 365/365 (100.0%) | Frontal pole, layer 1 |
| `parent_region` | str | 365/365 (100.0%) | FRP |
| `rgb_triplet` | list | 365/365 (100.0%) | [38, 143, 69] |
| `subregion_id` | int | 365/365 (100.0%) | 68 |
| `viz_layer` | int | 20/365 (5.5%) | 1 |
| `viz_x` | float | 20/365 (5.5%) | 61.80339887498945 |
| `viz_y` | float | 20/365 (5.5%) | -190.21130325903073 |

### Supertype

**数量**: 1,155

**属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `id` | int | 1155/1155 (100.0%) | 601 |
| `markers` | str | 1155/1155 (100.0%) | Car3,Nrg3,Gm34567 |
| `name` | str | 1155/1155 (100.0%) | 0001 CLA-EPd-CTX Car3 Glut_1 |
| `number_of_child_types` | int | 1155/1155 (100.0%) | 4 |
| `number_of_neurons` | int | 1155/1155 (100.0%) | 18474 |
| `tran_id` | int | 1155/1155 (100.0%) | 1 |
| `within_subclass_markers` | str | 1155/1155 (100.0%) | Itga8 |

## 关系类型

### AXON_NEIGHBOURING

**数量**: 128,575,491

**关系模式**:

- `(Neuron)-[AXON_NEIGHBOURING]->(Neuron)`: 128,575,491 (100.0%)

### BELONGS_TO

**数量**: 7,596

**关系模式**:

- `(Cluster)-[BELONGS_TO]->(Supertype)`: 5,204 (68.5%)
- `(Supertype)-[BELONGS_TO]->(Subclass)`: 1,155 (15.2%)
- `(ME_Subregion)-[BELONGS_TO]->(Subregion)`: 628 (8.3%)
- `(Subclass)-[BELONGS_TO]->(Class)`: 314 (4.1%)
- `(Subregion)-[BELONGS_TO]->(Region)`: 295 (3.9%)

### DEN_NEIGHBOURING

**数量**: 128,575,491

**关系模式**:

- `(Neuron)-[DEN_NEIGHBOURING]->(Neuron)`: 128,575,491 (100.0%)

### HAS_CLASS

**数量**: 2,391

**关系模式**:

- `(Subregion)-[HAS_CLASS]->(Class)`: 1,131 (47.3%)
- `(Region)-[HAS_CLASS]->(Class)`: 1,037 (43.4%)
- `(ME_Subregion)-[HAS_CLASS]->(Class)`: 223 (9.3%)

**关系属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `pct_cells` | float | 2391/2391 (100.0%) | 0.18140929535232383 |
| `rank` | int | 2391/2391 (100.0%) | 1 |

### HAS_CLUSTER

**数量**: 5,221

**关系模式**:

- `(Subregion)-[HAS_CLUSTER]->(Cluster)`: 2,670 (51.1%)
- `(Region)-[HAS_CLUSTER]->(Cluster)`: 2,017 (38.6%)
- `(ME_Subregion)-[HAS_CLUSTER]->(Cluster)`: 534 (10.2%)

**关系属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `pct_cells` | float | 5221/5221 (100.0%) | 0.07196401799100449 |
| `rank` | int | 5221/5221 (100.0%) | 1 |

### HAS_SUBCLASS

**数量**: 4,163

**关系模式**:

- `(Subregion)-[HAS_SUBCLASS]->(Subclass)`: 1,922 (46.2%)
- `(Region)-[HAS_SUBCLASS]->(Subclass)`: 1,839 (44.2%)
- `(ME_Subregion)-[HAS_SUBCLASS]->(Subclass)`: 402 (9.7%)

**关系属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `pct_cells` | float | 4163/4163 (100.0%) | 0.08395802098950525 |
| `rank` | int | 4163/4163 (100.0%) | 1 |

### HAS_SUPERTYPE

**数量**: 5,202

**关系模式**:

- `(Subregion)-[HAS_SUPERTYPE]->(Supertype)`: 2,565 (49.3%)
- `(Region)-[HAS_SUPERTYPE]->(Supertype)`: 2,148 (41.3%)
- `(ME_Subregion)-[HAS_SUPERTYPE]->(Supertype)`: 489 (9.4%)

**关系属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `pct_cells` | float | 5202/5202 (100.0%) | 0.07796101949025487 |
| `rank` | int | 5202/5202 (100.0%) | 1 |

### LOCATE_AT

**数量**: 18,857

**关系模式**:

- `(Neuron)-[LOCATE_AT]->(Region)`: 18,857 (100.0%)

### LOCATE_AT_ME_SUBREGION

**数量**: 9,312

**关系模式**:

- `(Neuron)-[LOCATE_AT_ME_SUBREGION]->(ME_Subregion)`: 9,312 (100.0%)

### LOCATE_AT_SUBREGION

**数量**: 20,322

**关系模式**:

- `(Neuron)-[LOCATE_AT_SUBREGION]->(Subregion)`: 20,322 (100.0%)

### NEIGHBOURING

**数量**: 1

**关系模式**:

- `(Neuron)-[NEIGHBOURING]->(Neuron)`: 1 (100.0%)

### PROJECT_TO

**数量**: 1,135,049

**关系模式**:

- `(Neuron)-[PROJECT_TO]->(Region)`: 591,345 (52.1%)
- `(Neuron)-[PROJECT_TO]->(Subregion)`: 487,825 (43.0%)
- `(Subregion)-[PROJECT_TO]->(Subregion)`: 20,570 (1.8%)
- `(Region)-[PROJECT_TO]->(Region)`: 18,087 (1.6%)
- `(Region)-[PROJECT_TO]->(Subregion)`: 17,222 (1.5%)

**关系属性**:

| 属性名 | 类型 | 覆盖率 | 样例值 |
|--------|------|--------|--------|
| `neuron_count` | int | 1135049/1135049 (100.0%) | 517 |
| `source_acronym` | str | 1135049/1135049 (100.0%) | ORBvl |
| `target_acronym` | str | 1135049/1135049 (100.0%) | ORBl |
| `total` | float | 1135049/1135049 (100.0%) | 2969416.9990381664 |
| `weight` | float | 1135049/1135049 (100.0%) | 5743.5531896289485 |
