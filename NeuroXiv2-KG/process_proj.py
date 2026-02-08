import time
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import os
import sys
import basicfunc
from features import utils_fe
import sys
import matplotlib as mpl
import glob
import scipy
import pickle
import SimpleITK as sitk

sys.path.append("..")
mpl.use('Agg')


class Neuron():
    '''
    return swc: pandas dataframe
    swc file header: ['id', 'type', 'x', 'y', 'z', 'radius', 'pid']
    eswc file header: ['id', 'type', 'x', 'y', 'z', 'radius', 'pid', 'seg_id', 'level', 'mode', 'timestamp', 'teraflyindex']
    soma: type=1 and pid=-1
    axon arbor : type=2
    dendritic arbor: type = [3,4]
    '''
    SWCHeader = ['id', 'type', 'x', 'y', 'z', 'radius', 'pid']
    ESWCHeader = ['id', 'type', 'x', 'y', 'z', 'radius', 'pid', 'seg_id', 'level', 'mode', 'timestamp', 'teraflyindex']
    NeuriteTypes = ['axon', 'dendrite', 'basal', 'apical']
    NeuColorDict = {'axon': 2, 'dendrite': 3, 'basal': 3, 'apical': 4}

    def __init__(self, path, retype=None, mode=None, scale=1, sep=' ', from_swc=None, prefind=False,flip_z=False):
        self.path = path
        self.swcname = os.path.split(path)[-1]
        if from_swc is None:
            self.df_swc = self.read_swc(self.path, sep=sep)
            # self.df_swc = self.read_swc(self.path, mode=mode, scale=scale, sep=sep)
        else:
            df_swc = pd.DataFrame(from_swc, dtype=np.float32, columns=self.SWCHeader)
            df_swc[["id", "type", "pid"]] = df_swc[["id", "type", "pid"]].astype(np.int32)
            self.df_swc = df_swc.copy()
            # self.df_swc = self.preprocess_swc(from_swc, mode=mode, scale=scale)
        if scale != 1:
            self.df_swc = self.swc_scale(scale=scale)
        if flip_z:
            self.df_swc = self.flip_z_coordinate()
        if retype is not None:
            self.df_swc = self.swc_retype(totype=retype)
        self.swc = self.df_swc.values.tolist()
        self.length = len(self.swc)
        self.neurite_types = self.num_of_neurite_type()
        if mode is not None:
            if mode.lower() == "all":
                self.arbors = [self.to_neurite_type(mode=neutype) for neutype in self.NeuriteTypes]
            else:
                self.df_swc = self.to_neurite_type(mode=mode.lower())
        if prefind:
            self.soma, self.bifurs, self.tips = [], [], []
            if self.length != 0:
                self.soma = self.df_swc.loc[self.get_soma(), self.SWCHeader].to_list()
                self.bifurs = self.df_swc[self.df_swc['id'].isin(self.get_bifs())].values.tolist()
                self.tips = self.df_swc[self.df_swc['id'].isin(self.get_tips())].values.tolist()

    def read_swc(self, file, sep=' ', header=None, skiprows=None):
        if skiprows is None:
            with open(file) as f:
                rows = f.read().splitlines()
            skiprows = 0
            for line in rows:
                if line[0] == "#":
                    skiprows += 1
                    continue
                else:
                    break
        swc = pd.read_csv(file, sep=sep, header=None, skiprows=skiprows)
        if header is None:
            header = self.SWCHeader
            if swc.shape[1] >= 12:
                header = self.ESWCHeader
                for i in np.arange(12, swc.shape[1]):
                    header.append('fea_' + str(i - 12))
        if len(header) == swc.shape[1]:
            swc.columns = header
        # swc.set_index(['id'], drop=True, inplace=True)
        return swc

    def save_swc(self, topath, inswc=None, header=None):
        if inswc is None:
            inswc = self.df_swc
        swc = inswc.copy()
        if header is None:
            header = self.SWCHeader
        else:
            for h in header:
                if h not in swc.keys().to_list():
                    header = swc.keys().to_list()
                    break
        # swc.reset_index(inplace=True)
        header[0] = '##' + header[0]
        swc.to_csv(topath, sep=' ', index=0, header=header)
        return True

    def warning_msg(self, msg):
        print("WARNING: file {0} {1}".format(self.path, msg))

    def to_neurite_type(self, inswc=None, mode=None):
        '''
        extract swc arbor :
            all arbors: mode="all", will return a list of swc_arbors order in ['axon','dendrite','basal','apical']
            axon: mode="axon"
            dendrite: mode="dendrite"
            apical dendrite: mode="apical"
            basal dendrite: mode="basal"
        return arbor swc
        '''
        if inswc is None:
            inswc = self.df_swc.copy()
        if mode is None or mode.lower() not in self.NeuriteTypes:
            self.warning_msg("Not a registered mode for neurite type: ['axon','dendrite','basal','apical']")
            return inswc
        target_types = [2, 3, 4]
        if mode.lower() == 'axon':
            target_types = [2]
        elif mode.lower() == 'dendrite':
            target_types = [3, 4]
        elif mode.lower() == 'basal':
            target_types = [3]
        elif mode.lower() == 'apical':
            target_types = [4]
        somaid = self.get_soma(inswc)
        # traversal from tip to soma
        target_tips = self.get_tips(inswc, ntype=target_types)
        if len(target_tips) == 0:
            self.warning_msg("No request arbor of " + mode.lower())
            return inswc
        swc = inswc.copy()
        swc.set_index(['id'], drop=True, inplace=True)
        target_ids = [somaid]
        for tip in target_tips:
            sid = tip
            target_ids.append(sid)
            spid = sid
            while True:
                spid = swc.loc[sid, 'pid']
                if spid == somaid or spid < 0:
                    break
                if spid in swc.index.to_list():
                    if spid not in target_ids:
                        target_ids.append(spid)
                    else:
                        break
                else:
                    self.warning_msg("Not possible")
                    break
                sid = spid
        outswc = inswc[inswc.id.isin(target_ids)].copy()
        return outswc

    def flip_z_coordinate(self, inswc=None, threshold=5700, total_height=11400):
        """
        翻转Z坐标：如果 z < threshold，则 z = total_height - z

        参数:
            inswc: 输入的swc DataFrame，默认为self.df_swc
            threshold: Z坐标阈值，默认5700
            total_height: 总高度，默认11400

        返回:
            翻转后的swc DataFrame
        """
        if inswc is None:
            inswc = self.df_swc.copy()

        if inswc.shape[0] == 0:
            return inswc

        # 创建需要翻转的掩码
        flip_mask = inswc['z'] < threshold

        # 翻转满足条件的Z坐标
        inswc.loc[flip_mask, 'z'] = total_height - inswc.loc[flip_mask, 'z']

        return inswc

    def num_of_neurite_type(self, inswc=None):
        if inswc is None:
            inswc = self.df_swc.copy()
        tipids = self.get_tips(inswc)
        tips = inswc[inswc.id.isin(tipids)].copy()
        return tips['type'].value_counts().index.tolist()

    def swc_scale(self, inswc=None, scale=1):
        '''scale the coordinate of swc'''
        if inswc is None:
            inswc = self.df_swc.copy()
        if inswc.shape[0] == 0 or scale == 1:
            return inswc
        inswc['x'] *= np.float16(scale)
        inswc['y'] *= np.float16(scale)
        inswc['z'] *= np.float16(scale)
        return inswc

    def swc_retype(self, inswc=None, totype=2, rid=None):
        '''change type of swc'''
        if inswc is None:
            inswc = self.df_swc.copy()
        if inswc.shape[0] == 0 or totype <= 1:
            return inswc
        if rid is None:
            rid = self.get_soma(inswc)
        inswc['type'] = totype
        inswc.loc[rid, 'type'] = 1
        return inswc

    def get_degree(self, inswc=None):
        '''区分不同类型的node
        internode: degree =2
        tipnode: degree =1
        bifurcation: degree =3
        soma node: degree >=3
        '''
        if inswc is None:
            tswc = self.df_swc.copy()
        else:
            tswc = inswc.copy()
        tswc.set_index(['id'], drop=True, inplace=True)
        tswc['degree'] = tswc['pid'].isin(tswc.index).astype('int')
        # print(tswc['degree'])
        n_child = tswc.pid.value_counts()
        n_child = n_child[n_child.index.isin(tswc.index)]
        tswc.loc[n_child.index, 'degree'] = tswc.loc[n_child.index, 'degree'] + n_child
        return tswc

    def get_soma(self, inswc=None):
        '''return index of soma node in df_swc'''
        if inswc is None:
            df = self.df_swc.copy()
        else:
            df = inswc.copy()
        df_soma = df[(df["type"] == 1) & (df["pid"] == -1)].copy()
        if df_soma.shape[0] == 0:
            df_soma = df[df["pid"] == -1].copy()
            if df_soma.shape[0] == 0:
                self.warning_msg("No soma (type=1) detected...try to find root(pid=-1)")
                self.warning_msg("No soma detected...")
                return None
        if df_soma.shape[0] > 1:
            self.warning_msg("multiple soma detected.")
            return None
        # return df_soma.values[0].tolist()
        return df_soma.index[0]

    def get_keypoints(self, inswc=None, rid=None):
        '''
        Key points: soma,bifurcations,tips
        return idlist
        '''
        if inswc is None:
            swc = self.df_swc.copy()
        else:
            swc = inswc.copy()
        if rid is None:
            rid = self.get_soma(inswc)
        # print(swc.shape)
        swc = self.get_degree(swc)
        idlist = swc[((swc.degree != 2) | (swc.index == rid))].index.tolist()
        return idlist

    def get_tips(self, inswc=None, ntype=None):
        if inswc is None:
            swc = self.df_swc.copy()
        else:
            swc = inswc.copy()
        if swc.shape[0] == 0:
            return None
        swc = self.get_degree(swc)
        if ntype is not None:
            idlist = swc[(swc.degree < 2) & (swc.type.isin(ntype))].index.tolist()
        else:
            idlist = swc[(swc.degree < 2)].index.tolist()
        return idlist

    def get_bifs(self, inswc=None, rid=None, ntype=None):
        if inswc is None:
            swc = self.df_swc.copy()
        else:
            swc = inswc.copy()
        if rid is None:
            rid = self.get_soma(swc)
        # print(swc.shape)
        swc = self.get_degree(swc)
        if ntype is not None:
            idlist = swc[((swc.degree > 2) & (swc.index != rid) & (swc.type == ntype))].index.tolist()
        else:
            idlist = swc[((swc.degree > 2) & (swc.index != rid))].index.tolist()
        return idlist

    def swc2branches(self, inswc=None):
        '''
        reture branch list of a swc
        branch: down to top
        '''
        if inswc is None:
            inswc = self.df_swc.copy()
        keyids = self.get_keypoints(inswc)
        branches = []
        for key in keyids:
            if inswc.loc[key, 'pid'] < 0 | inswc.loc[key, 'type'] <= 1:
                continue
            branch = []
            branch.append(key)
            pkey = inswc.loc[key, 'pid']
            while True:
                branch.append(pkey)
                if pkey in keyids:
                    break
                key = pkey
                pkey = inswc.loc[key, 'pid']
            branches.append(branch)
        return branches

    def save_swc_old(self, path, comments='', eswc=False):
        '''
        save swc file
        :param path:save path
        :param swc:swc list
        :param comments:some remarks in line 2 in swc file
        :return:none
        '''
        if not path.endswith(".swc"):
            path += ".swc"
        with open(path, 'w') as f:
            f.writelines('#' + comments + "\n")
            f.writelines("#n,type,x,y,z,radius,parent\n")
            for node in self.swc:
                string = ""
                for i in range(len(node)):
                    item = node[i]
                    if i in [0, 1, 6]:
                        item = int(item)
                    elif i in [2, 3, 4, 5]:
                        item = f'{item:.3f}'

                    string = string + str(item) + " "
                    if not eswc:
                        if i == 6:
                            break
                string = string.strip(" ")
                string += "\n"
                f.writelines(string)


def alignment(neuron: Neuron):
    n = utils_fe.Neuron()
    n.load_eswc(neuron.swc)
    n.normalize_neuron(ntype=list(set(n.ntype)), dir_order='zyx')
    nn = n.convert_to_swclist()
    return Neuron(path=neuron.path, from_swc=nn, prefind=True)


class SWC_Features():
    '''
    some new features. (see feature_name)
    (swc need resample step=10!!!)
    '''

    def __init__(self, neuron: Neuron, swc_reg=None):
        self.neuron = neuron
        self.feature_name = ["Center Shift", "Relative Center Shift",
                             "Average Contraction", "Average Bifurcation Angle Remote",
                             "Average Bifurcation Angle Local",
                             "Max Branch Order", "Number of Bifurcations", "Total Length",
                             "Max Euclidean Distance", "Max Path Distance", "Average Euclidean Distance",
                             "25% Euclidean Distance",
                             "50% Euclidean Distance", "75% Euclidean Distance", "Average Path Distance",
                             "25% Path Distance",
                             "50% Path Distance", "75% Path Distance",
                             '2D Density', '3D Density',
                             'Area', 'Volume', 'Width', 'Width_95ci', 'Height', 'Height_95ci', 'Depth', 'Depth_95ci',
                             'Slimness', 'Slimness_95ci', 'Flatness', 'Flatness_95ci']
        self.feature_dict = {}
        for fn in self.feature_name:
            self.feature_dict[fn] = None

        if self.neuron.length == 0:
            return

        self.swc = neuron.swc
        self.path = neuron.path
        self.soma = self.neuron.soma
        self.tips = self.neuron.tips
        self.swc_reg = swc_reg
        # self.bifurs = get_bifurs(swc)

        self.calc_feature()

    def calc_feature(self):
        # self.Euc_Dis(self.swc)
        self.Pat_Dis_xuan(self.swc)
        self.center_shift(self.swc)
        if self.feature_dict["Max Euclidean Distance"] is None:
            self.feature_dict["Relative Center Shift"] = None
        else:
            self.feature_dict["Relative Center Shift"] = self.feature_dict["Center Shift"] / self.feature_dict[
                "Max Euclidean Distance"]
        self.size_related_features(self.swc)
        self.xyz_approximate(self.swc)
        self.feature_dict["Number of Bifurcations"] = len(self.neuron.bifurs)

    def Euc_Dis(self, swc):
        dislist = cdist(np.array([self.soma[2:5]]), np.array(swc)[:, 2:5])[0]
        if len(dislist) == 0:
            return [None] * 5
        euc_dis_ave = np.mean(dislist)
        euc_dis_max = np.max(dislist)
        euc_dis_25, euc_dis_50, euc_dis_75 = np.percentile(dislist, [25, 50, 75])

        self.feature_dict["Max Euclidean Distance"] = euc_dis_max
        self.feature_dict["Average Euclidean Distance"] = euc_dis_ave
        self.feature_dict["25% Euclidean Distance"] = euc_dis_25
        self.feature_dict["50% Euclidean Distance"] = euc_dis_50
        self.feature_dict["75% Euclidean Distance"] = euc_dis_75

        return

    def Pat_Dis_xuan(self, swc):
        NT = utils_fe.NeuronTree()
        NT.readSwc_fromlist(swc)
        NT.computeFeature()
        pathdislist = NT.pathTotal
        euxdislist = NT.euxTotal
        length = len(pathdislist)
        if len(pathdislist) == 0 or len(euxdislist) == 0:
            return [None] * 5
        path_dis_ave = np.mean(pathdislist)
        path_dis_max = np.max(pathdislist)
        path_dis_25, path_dis_50, path_dis_75 = np.percentile(pathdislist, [25, 50, 75])

        euc_dis_ave = np.mean(euxdislist)
        euc_dis_25, euc_dis_50, euc_dis_75 = np.percentile(euxdislist, [25, 50, 75])

        self.feature_dict["Max Path Distance"] = path_dis_max
        self.feature_dict["Average Path Distance"] = path_dis_ave
        self.feature_dict["25% Path Distance"] = path_dis_25
        self.feature_dict["50% Path Distance"] = path_dis_50
        self.feature_dict["75% Path Distance"] = path_dis_75
        self.feature_dict["Total Length"] = NT.Length
        self.feature_dict["Average Contraction"] = NT.Contraction
        self.feature_dict["Average Bifurcation Angle Remote"] = NT.BifA_remote
        self.feature_dict["Average Bifurcation Angle Local"] = NT.BifA_local
        self.feature_dict["Max Branch Order"] = NT.Max_Order
        self.feature_dict["Max Euclidean Distance"] = NT.Max_Eux
        self.feature_dict["Average Euclidean Distance"] = euc_dis_ave
        self.feature_dict["25% Euclidean Distance"] = euc_dis_25
        self.feature_dict["50% Euclidean Distance"] = euc_dis_50
        self.feature_dict["75% Euclidean Distance"] = euc_dis_75

    def Pat_Dis(self, swc):
        patlist = []
        soma = self.soma
        id_pathdist = {}
        idlist = np.array(swc)[:, 0].tolist()
        pidlist = np.array(swc)[:, 6].tolist()
        if soma[0] not in pidlist:
            # 此时说明没有连接到soma的通路，寻找最接近soma的root
            maxdist = 1000000
            for node in swc:
                if node == self.soma:
                    continue
                if node[6] not in idlist:
                    cur_dist = np.linalg.norm(np.array(self.soma[2:5]) - np.array(node[2:5]), ord=2)
                    if cur_dist < maxdist:
                        maxdist = cur_dist
                        soma = node

        if self.tips:
            nodes = self.tips
        else:
            nodes = swc
        for node in nodes:
            if node == soma or node == self.soma:
                continue
            cur_node = node
            cur_pathdist = 0
            passbynode = {}
            while True:
                pid = cur_node[6]
                if pid not in idlist:
                    break
                idx = idlist.index(pid)
                new_node = swc[idx]
                delta_pathdist = np.linalg.norm(np.array(cur_node[2:5]) - np.array(new_node[2:5]), ord=2)
                cur_pathdist += delta_pathdist
                if passbynode.keys():
                    passbynode = dict(
                        zip(list(passbynode.keys()), (np.array(list(passbynode.values())) + delta_pathdist).tolist()))
                if new_node == soma:
                    id_pathdist[node[0]] = cur_pathdist
                    id_pathdist.update(passbynode)
                    break
                elif new_node[0] in id_pathdist.keys():
                    id_pathdist[node[0]] = cur_pathdist + id_pathdist[new_node[0]]
                    if passbynode.keys():
                        passbynode = dict(
                            zip(list(passbynode.keys()),
                                (np.array(list(passbynode.values())) + id_pathdist[new_node[0]]).tolist()))
                    id_pathdist.update(passbynode)
                    break
                else:
                    cur_node = new_node
                    passbynode[new_node[0]] = 0

        pathdislist = list(id_pathdist.values())
        length = len(pathdislist)
        if length == 0:
            return [None] * 5
        path_dis_ave = np.mean(pathdislist)
        path_dis_max = np.max(pathdislist)
        path_dis_25, path_dis_50, path_dis_75 = np.percentile(pathdislist, [25, 50, 75])

        self.feature_dict["Max Path Distance"] = path_dis_max
        self.feature_dict["Average Path Distance"] = path_dis_ave
        self.feature_dict["25% Path Distance"] = path_dis_25
        self.feature_dict["50% Path Distance"] = path_dis_50
        self.feature_dict["75% Path Distance"] = path_dis_75

        return

    def center_shift(self, swc):
        soma = self.soma
        swc_ar = np.array(swc)
        centroid = np.mean(swc_ar[:, 2:5], axis=0)
        self.feature_dict["Center Shift"] = np.linalg.norm(np.array(soma[2:5]) - centroid[0:3], ord=2)
        return

    def pixel_voxel_calc(self, swc_xyz):
        swcxyz = np.array(swc_xyz)
        x = np.round(swcxyz[:, 0])
        y = np.round(swcxyz[:, 1])
        z = np.round(swcxyz[:, 2])
        pixels = list(set(list(zip(z, y))))  # 投射到zy平面算pixel z是主方向 且去除了冗余pixel
        voxels = list(set(list(zip(x, y, z))))
        num_pixels = len(pixels)
        num_voxels = len(voxels)

        return num_pixels, num_voxels

    def size_related_features(self, swc):
        num_nodes = len(swc)
        if num_nodes <= 3:
            return [None] * 4
        swc_zy = np.array(swc)[:, 3:5]
        swc_xyz = np.array(swc)[:, 2:5]

        try:
            CH2D = ConvexHull(swc_zy)
            CH3D = ConvexHull(swc_xyz)
        except scipy.spatial.qhull.QhullError:
            return [None] * 4
        # CH2D.area  # 2D情况下这个是周长×
        # CH2D.volume  # 2D情况下这个是面积√
        # CH3D.area    # 3D情况下这个是表面积×
        # CH3D.volume  # 3D情况下这个是体积√
        area = CH2D.volume
        volume = CH3D.volume
        # interpolation of swc so that each pixel/voxel can be occupied on all pathway
        swc_xyz_new = list(swc_xyz)
        swc_arr = np.array(swc)
        idlist = list(swc_arr[:, 0])
        for node in swc:
            pid = node[6]
            x1, y1, z1 = node[2:5]
            if pid not in idlist:
                continue
            else:
                cur_id = idlist.index(pid)
                x2, y2, z2 = swc_xyz[cur_id]
                count = int(np.linalg.norm([x1 - x2, y1 - y2, z1 - z2]) // 1)
                if count != 0:
                    tmp = [[x1 + 1 * x, y1 + 1 * x, z1 + 1 * x] for x in
                           range(1, count + 1)]
                    swc_xyz_new.extend(tmp)

        num_pixels, num_voxels = self.pixel_voxel_calc(swc_xyz_new)
        density_2d = num_pixels / area
        density_3d = num_voxels / volume
        self.feature_dict["Area"] = area
        self.feature_dict["Volume"] = volume
        self.feature_dict["2D Density"] = density_2d
        self.feature_dict["3D Density"] = density_3d

        return

    def xyz_approximate(self, swc):
        '''
        shape related
        :param swc:
        :return:
        '''
        if not swc:
            return [None] * 10
        swcxyz = np.array(swc)[:, 2:5]
        x = swcxyz[:, 0]
        y = swcxyz[:, 1]
        z = swcxyz[:, 2]
        width = np.max(y) - np.min(y)  # y  zyx-registration   height=z-z' width=y-y' depth=x-x'
        height = np.max(z) - np.min(z)  # z
        depth = np.max(x) - np.min(x)  # x
        # confidence interval 95%
        width_95ci = abs(np.percentile(y, 97.5) - np.percentile(y, 2.5))
        height_95ci = abs(np.percentile(z, 97.5) - np.percentile(z, 2.5))
        depth_95ci = abs(np.percentile(x, 97.5) - np.percentile(x, 2.5))

        slimness = width / height  # slimness = width/height
        flatness = height / depth  # flatness = height/depth
        slimness_95ci = width_95ci / height_95ci
        flatness_95ci = height_95ci / depth_95ci

        self.feature_dict["Width"] = width
        self.feature_dict["Height"] = height
        self.feature_dict["Depth"] = depth
        self.feature_dict["Width_95ci"] = width_95ci
        self.feature_dict["Height_95ci"] = height_95ci
        self.feature_dict["Depth_95ci"] = depth_95ci
        self.feature_dict["Slimness"] = slimness
        self.feature_dict["Flatness"] = flatness
        self.feature_dict["Slimness_95ci"] = slimness_95ci
        self.feature_dict["Flatness_95ci"] = flatness_95ci

        return


def Pipeline(file_path, retype=None):
    fp_split = os.path.split(file_path)
    folder_path = fp_split[0]
    swcname = fp_split[-1]
    swcpath = os.path.join(folder_path, swcname)
    # print('a')
    n_axon = Neuron(swcpath,flip_z=True)
    if 'CCFv3' in swcname:
        proj_dict_axon = basicfunc.calc_projection(n_axon.swc, anno=anno_ME)
    # elif 'CCF-thin' in swcname:
    #     proj_dict_axon = basicfunc.calc_projection(n_axon.swc, anno=anno_fmost)
    # print('d')
    neu_re4_align_axon = alignment(n_axon)
    # print('e')
    n_axon_re4_align_fe = SWC_Features(neu_re4_align_axon)
    # print('f')
    return n_axon_re4_align_fe, proj_dict_axon

def Projection_Pipeline(file_path, retype=None):
    fp_split = os.path.split(file_path)
    folder_path = fp_split[0]
    swcname = fp_split[-1]
    swcpath = os.path.join(folder_path, swcname)
    # print('a')
    n_axon = Neuron(swcpath,flip_z=True)
    if 'CCFv3' in swcname:
        proj_dict_axon = basicfunc.calc_projection(n_axon.swc, anno=anno_ME)
    # elif 'CCF-thin' in swcname:
    #     proj_dict_axon = basicfunc.calc_projection(n_axon.swc, anno=anno_fmost)
    return proj_dict_axon

def process_file(filepath, topath):
    filename = os.path.split(filepath)[-1]
    print(filename, end='\t')

    b_axon, proj_dict_axon = Pipeline(filepath, retype=None)
    bv_axon = list(b_axon.feature_dict.values())
    # bv_axon
    this_pf = os.path.join(topath, filename + '.pickle')
    if not os.path.exists(this_pf):
        with open(this_pf, 'wb') as f:
            pickle.dump((bv_axon, list(proj_dict_axon.values())), f)
    print('finished <<< ')

def process_file_ME_Proj(filepath, topath):
    filename = os.path.split(filepath)[-1]
    print(filename, end='\t')

    proj_dict_axon = Projection_Pipeline(filepath, retype=None)
    # bv_axon
    this_pf = os.path.join(topath, filename + '.pickle')
    if not os.path.exists(this_pf):
        with open(this_pf, 'wb') as f:
            pickle.dump(list(proj_dict_axon.values()), f)
    print('finished <<< ')

def loal_pickles(filepath):
    pfilename = os.path.split(filepath)[-1]
    filename = pfilename.split('.pick')[0]
    print(filename, end='\t')
    with open(filepath, 'rb') as f:
        bv_axon, proj_dict_axon = pickle.load(f)
        bvl_axon.append(bv_axon)
        projl_axon.append(proj_dict_axon)
        indice.append(filename.split('.sw')[0])
    print('finished <<< ')

def loal_pickles_ME(filepath):
    pfilename = os.path.split(filepath)[-1]
    filename = pfilename.split('.pick')[0]
    print(filename, end='\t')
    with open(filepath, 'rb') as f:
        proj_dict_axon = pickle.load(f)
        projl_axon.append(proj_dict_axon)
        indice.append(filename.split('.sw')[0])
    print('finished <<< ')


if __name__ == "__main__":

    annotmp = sitk.GetArrayFromImage(sitk.ReadImage(
        "/home/wlj/NeuroXiv2/data/annotation_25.nrrd"))
    # annotmp_fmost = sitk.GetArrayFromImage(sitk.ReadImage("/home/penglab/GitRepo/neuroxiv_api/testdata/annotation_25_fmost.nrrd"))
    annotmp_ME = sitk.GetArrayFromImage(sitk.ReadImage(
        "/home/wlj/NeuroXiv2/data/parc_r671_full.nrrd"))
    anno = np.transpose(annotmp, axes=[2, 1, 0])
    # anno_fmost = np.transpose(annotmp_fmost, axes=[2, 1, 0])
    anno_ME = np.transpose(annotmp_ME, axes=[2, 1, 0])


    # ========== 在这里直接指定输入输出路径 ==========
    input_path = "/home/wlj/NeuroXiv2/data/SWC/all_swc"  # 输入文件夹路径
    output_path = "/home/wlj/NeuroXiv2/data/SWC/pickles"  # 输出文件夹路径
    # ===============================================

    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 处理单个文件
    if os.path.isfile(input_path) and input_path.endswith('.swc'):
        print(f'Processing single file: {input_path}')
        # process_file(input_path, output_path)
        process_file_ME_Proj(input_path, output_path)
        print('Processing completed!')

    # 批量处理文件夹
    elif os.path.isdir(input_path):
        import concurrent.futures

        # 获取所有SWC文件
        all_swc_files = glob.glob(os.path.join(input_path, '*.swc'))
        print(f'Found {len(all_swc_files)} SWC files in {input_path}')

        if len(all_swc_files) == 0:
            print('No SWC files found!')
            exit(1)

        # 多线程批量处理
        num_threads = 40
        print(f'Starting batch processing with {num_threads} threads...')


        def process_wrapper(swc_file):
            try:
                process_file_ME_Proj(swc_file, output_path)
                return True
            except Exception as e:
                print(f'Error processing {swc_file}: {str(e)}')
                return False


        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_wrapper, all_swc_files))

        success_count = sum(results)
        print(f'\nBatch processing completed!')
        print(f'Successfully processed: {success_count}/{len(all_swc_files)} files')
        print(f'Output saved to: {output_path}')

    else:
        print(f'Error: Invalid input path: {input_path}')
        print('Input should be either a .swc file or a directory containing .swc files')
        exit(1)

    # # Step1: generate neuron features from each neuron
    # swcfile = sys.argv[1]
    # todir = sys.argv[2]
    # # print(swcfile,todir)
    # if os.path.exists(swcfile) and os.path.exists(todir):
    #     print('start to process....', end='\t')
    #     process_file(swcfile, todir)
    # exit(0)




    # Step2: load neuron features from file

    from tornado import concurrent
    # pickle_path = rf"/home/penglab/Data/my_dataserver/NeuroXiv_datasets/features/axonpickles"
    # pickle_path = rf"/home/penglab/Data/my_dataserver/NeuroXiv_datasets/features/denpickles"
    pickle_path = rf"/home/wlj/NeuroXiv2/data/SWC/pickles"
    fpath = rf"/home/wlj/NeuroXiv2/data/SWC/all_swc/ION_full_210014_012_CCFv3.swc"
    allpickles = glob.glob(os.path.join(pickle_path, '*.pickle'))
    print('ALL=', len(allpickles), allpickles[0])
    #
    bvl = []
    projl = []
    bvl_axon = []
    projl_axon = []
    indice = []
    b_axon, proj_dict_axon = Pipeline(fpath, retype=None)

    num_threads = 40
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(loal_pickles_ME, allpickles)
    # dfb_axon = pd.DataFrame(bvl_axon, index=indice, columns=list(b_axon.feature_dict.keys()))
    # dfb_axon.index.name='ID'
    # dfb_axon.to_csv("/home/penglab/Data/my_dataserver/NeuroXiv_datasets/neuFeatures/features_v202501/axonfull_morpho.csv")
    # dfb_axon.to_csv("/home/penglab/Data/my_dataserver/NeuroXiv_datasets/features/local_morpho.csv")
    # dfb_axon.to_csv("/home/penglab/Data/my_dataserver/NeuroXiv_datasets/neuFeatures/features_v202501/denfull_morpho.csv")

    df_proj_axon = pd.DataFrame(projl_axon, index=indice, columns=list(proj_dict_axon.keys()))
    df_proj_axon.index.name='ID'
    df_proj_axon.to_csv("/home/wlj/NeuroXiv2/data/SWC/axonfull_proj_ME.csv")
    # df_proj_axon.to_csv("/home/penglab/Data/my_dataserver/NeuroXiv_datasets/neuFeatures/features_v202501/denfull_proj.csv")
    # df_proj_axon.to_csv("/home/penglab/Data/my_dataserver/NeuroXiv_datasets/neuFeatures/features_v202501/localfull_proj.csv")
