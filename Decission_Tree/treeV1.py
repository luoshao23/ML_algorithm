import numpy as np
import pandas as pd
import time


class decisionnode(object):
    """Define a basic class: node"""

    def __init__(self, col=-1, id_=None, tb=None, fb=None, p=None, size=None, F1=None, F2=None, attrs=None,  set_=None):
        """
        node instance init function
        """
        self.col = col            # feature selected
        self.id_ = id_            # node id
        self.tb = tb              # left node
        self.fb = fb              # right node
        self.p = p                # parent node
        self.size = size          # pop size in this node
        self.F1 = F1              # Metric F1
        self.F2 = F2              # Metric F2
        self.attr_list = attrs    # alternative feature list
        self.set_ = set_          # pop set for left/right node

    def split(self, attributes):
        df = self.set_
        df_size = len(df)
        print "spliting %d" % self.id_

        # init a metric result dict
        info = pd.DataFrame(index=attributes, columns=[
                            "C2", "C3", "C4", "C6", "C7", "F11", "F10", "F2"])

        # estimate each attr and stored in info
        for attr in attributes:
            info.loc[attr] = check_search(df, attr)
        print "Attr search loop done"

        # filter qualified info
        info = info[(info[["C2", "C3", "C4", "C6", "C7"]] == 1).all(axis=1)]
        info.sort_values(by="F2", ascending=0, inplace=True)

        #if no feature retained return None
        if info.empty:
            self.update_node(size=df_size)
            print '[Empty] No feature qualified'
            return None
        selected_attrs = info.iloc[:]

        # choose top 1 attribute
        top_attr = selected_attrs.iloc[0]

        # if F2>0 do regular split
        if top_attr.F2 > 0:
            print "Node %d successful" % self.id_
            V = top_attr.name
            df = self.set_
            left = df[df[V] == 1]
            right = df[df[V] == 0]

            left_node = decisionnode(F1=top_attr.F11, size=len(
                left), id_=self.id_ * 2, p=self, set_=left)
            right_node = decisionnode(F1=top_attr.F10, size=len(
                right), id_=self.id_ * 2 + 1, p=self, set_=right)
            self.update_node(col=V, size=df_size, F2=top_attr.F2,
                             tb=left_node, fb=right_node, attrs=selected_attrs)
            print self.tb.F1, self.fb.F1
            return [left_node, right_node]
        else:
        # if F2==0 process is identical while the result probably makes no business sense
            print "F2=0 at %d, split node randomly" % self.id_
            V = top_attr.name
            df = self.set_
            left = df[df[V] == 1]
            right = df[df[V] == 0]

            left_node = decisionnode(F1=top_attr.F11, size=len(
                left), id_=self.id_ * 2, p=self, set_=left)
            right_node = decisionnode(F1=top_attr.F10, size=len(
                right), id_=self.id_ * 2 + 1, p=self, set_=right)
            self.update_node(col=V, size=df_size, F2=top_attr.F2,
                             tb=left_node, fb=right_node, attrs=selected_attrs)
            print self.tb.F1, self.fb.F1
            return [left_node, right_node]

    def update_node(self, **kwargs):
        """for updating node while splitting"""
        if "col" in kwargs:
            self.col = kwargs["col"]
        if "id_" in kwargs:
            self.id_ = kwargs["id_"]
        if "tb" in kwargs:
            self.tb = kwargs["tb"]
        if "fb" in kwargs:
            self.fb = kwargs["fb"]
        if "size" in kwargs:
            self.size = kwargs["size"]
        if "F1" in kwargs:
            self.F1 = kwargs["F1"]
        if "F2" in kwargs:
            self.F2 = kwargs["F2"]
        if "attrs" in kwargs:
            self.attr_list = kwargs["attrs"]
        if "set_" in kwargs:
            self.set_ = kwargs["set_"]

    def print_node(self):
        """visualization"""
        if self.col != -1:
            print "%4d %5.3f %8d %20s %10d %10d" % (self.id_, self.F1, self.size, self.col, self.tb.id_, self.fb.id_)
            self.tb.print_node()
            self.fb.print_node()
        else:
            print "%4d %5.3f %8d %20s Null Null" % (self.id_, self.F1, self.size, self.col)


def clean_data(DT_df):
    """data preprocessing"""
    DT_df = DT_df.drop(drop_cols, axis=1)
    DT_df["fs_scan_amt_pre"] = DT_df["fs_scan_amt_pre"].astype(float)
    DT_df["fs_scan_amt_pos"] = DT_df["fs_scan_amt_pos"].astype(float)
    DT_df["fs_scan_amt_pos_PF"] = DT_df["fs_scan_amt_pos_PF"].astype(float)
    DT_df["dyn_margin_amt_pre"] = DT_df["dyn_margin_amt_pre"].astype(float)
    DT_df["dyn_margin_amt_pos"] = DT_df["dyn_margin_amt_pos"].astype(float)
    DT_df["dyn_margin_amt_pos_PF"] = DT_df[
        "dyn_margin_amt_pos_PF"].astype(float)
    DT_df["ctl_grp_ind"] = DT_df["ctl_grp_ind"].astype(int)
    DT_df["mailer_version_id"] = DT_df["mailer_version_id"].astype(int)
    DT_df["tcm_redeem_md"] = pd.to_numeric(DT_df["tcm_redeem_md"])
    for attr in attributes:
        DT_df[attr] = DT_df[attr].astype(int)

    fields = attributes + ["fs_scan_amt_pre", "fs_scan_amt_pos", "fs_scan_amt_pos_PF", "dyn_margin_amt_pre", "dyn_margin_amt_pos", "dyn_margin_amt_pos_PF",
                           "ctl_grp_ind", "mailer_version_id", "tcm_redeem_md", "xtra_card_nbr"]
    DT_df = DT_df[fields]
    return DT_df


def agg_by_attr(df):
    """aggregate data by `ctl_grp_ind` and return `result`"""
    result = pd.Series()
    df_CT = df.groupby(["ctl_grp_ind"]).agg({"xtra_card_nbr": "count", "fs_scan_amt_pre": "sum", "fs_scan_amt_pos": "sum",
                                             "fs_scan_amt_pos_PF": "sum", "dyn_margin_amt_pre": "sum", "dyn_margin_amt_pos": "sum",
                                             "dyn_margin_amt_pos_PF": "sum"})
    if df_CT.shape != (2, 7):
        return None

    result["cnt_card_T"] = df_CT.loc[0]["xtra_card_nbr"]
    result["cnt_card_C"] = df_CT.loc[1]["xtra_card_nbr"]
    result["fs_scan_amt_pre_T"] = df_CT.loc[0]["fs_scan_amt_pre"]
    result["fs_scan_amt_pre_C"] = df_CT.loc[1]["fs_scan_amt_pre"]
    result["fs_scan_amt_pos_T"] = df_CT.loc[0]["fs_scan_amt_pos"]
    result["fs_scan_amt_pos_C"] = df_CT.loc[1]["fs_scan_amt_pos"]
    result["fs_scan_amt_pos_PF_T"] = df_CT.loc[0]["fs_scan_amt_pos_PF"]
    result["fs_scan_amt_pos_PF_C"] = df_CT.loc[1]["fs_scan_amt_pos_PF"]
    result["dyn_margin_amt_pre_T"] = df_CT.loc[0]["dyn_margin_amt_pre"]
    result["dyn_margin_amt_pre_C"] = df_CT.loc[1]["dyn_margin_amt_pre"]
    result["dyn_margin_amt_pos_T"] = df_CT.loc[0]["dyn_margin_amt_pos"]
    result["dyn_margin_amt_pos_C"] = df_CT.loc[1]["dyn_margin_amt_pos"]
    result["dyn_margin_amt_pos_PF_T"] = df_CT.loc[0]["dyn_margin_amt_pos_PF"]
    result["dyn_margin_amt_pos_PF_C"] = df_CT.loc[1]["dyn_margin_amt_pos_PF"]
    # result["cnt_redeemer"]=df[np.logical_and(df.ctl_grp_ind==0,df.ind_tcm_redeem==1)].count()
    result["TCM_MD"] = df[df.ctl_grp_ind == 0]["tcm_redeem_md"].sum()

    result["GSL"] = (result.fs_scan_amt_pos_T - result.fs_scan_amt_pre_T *
                     result.fs_scan_amt_pos_C / result.fs_scan_amt_pre_C) / result.cnt_card_T
    result["NMI"] = (0.4 * (result.fs_scan_amt_pos_T - result.fs_scan_amt_pre_T *
                            result.fs_scan_amt_pos_C / result.fs_scan_amt_pre_C) + result.TCM_MD) / result.cnt_card_T
    result["NMI_PF"] = (0.4 * (result.fs_scan_amt_pos_PF_T - result.fs_scan_amt_pre_T *
                               result.fs_scan_amt_pos_PF_C / result.fs_scan_amt_pre_C) + result.TCM_MD) / result.cnt_card_T
    result["DYN"] = (result.dyn_margin_amt_pos_T - result.dyn_margin_amt_pre_T *
                     result.dyn_margin_amt_pos_C / result.dyn_margin_amt_pre_C) / result.cnt_card_T
    result["DYN_PF"] = (result.dyn_margin_amt_pos_PF_T - result.dyn_margin_amt_pre_T *
                        result.dyn_margin_amt_pos_PF_C / result.dyn_margin_amt_pre_C) / result.cnt_card_T
    result["PRE_VARIANCE"] = (result.fs_scan_amt_pre_T / result.cnt_card_T - result.fs_scan_amt_pre_C /
                              result.cnt_card_C) / (result.fs_scan_amt_pre_C / result.cnt_card_C)
    # result["cnt_card_all"]=1.5*result.cnt_card_T
    result["cnt_card_all"] = result.cnt_card_T + result.cnt_card_C
    result["cnt_card_C"] = result.cnt_card_C
    result["Metric"] = (result.NMI_PF + result.DYN_PF) / 2   # main metric for node splitting
    return result


def check_search(df, var):
    """aggregate data by a certain feature"""
    info = pd.Series()
    g = df.groupby(var).apply(agg_by_attr)
#     print g
    if not (g.shape[0] == 2 and isinstance(g.iloc[0], pd.Series)):
        #         print "Bad aggregation"
        return info
    info["C2"] = 1 if all(abs(g.PRE_VARIANCE < 0.05)) else 0
    info["C3"] = 1 if all(1.1 * g.NMI > g.NMI_PF) else 0
    info["C4"] = 1 if all(1.1 * g.DYN > g.DYN_PF) else 0
    info["C6"] = 1 if all(g.cnt_card_all > 40000) else 0
    info["C7"] = 1 if all(g.cnt_card_C > 5000) else 0
    info["F11"] = g.loc[1].Metric
    info["F10"] = g.loc[0].Metric
    info["F2"] = abs(info["F11"] - info["F10"])
    return info


def search_Metric(this, ind):
    """auxiliary function for seeking value of F1 given a node id"""
    if this.col != -1:
        return max(search_Metric(this.tb, ind), search_Metric(this.fb, ind))
    else:
        if this.id_ == ind:
            return this.F1
        else:
            return None


class myTree(object):
    """tree class for dual training"""

    def __init__(self, root=None, attributes=None):
        self.root = None
        self.attributes = attributes

    def dual_train(self, df, vdfs):
        attributes = self.attributes
        # make a root node for a tree
        if not self.root:
            self.root = decisionnode(id_=1, set_=df)
            self.root.F1 = agg_by_attr(df).Metric
            nodes = self.root.split(attributes)
        # bfs algorithm, do while there is any node at this level
        while nodes:
            this_level = nodes
            nodes = []
            print "this level: %r" % this_level
            # loop in all nodes at this level
            for node in this_level:
                # split this node
                ns = node.split(attributes)
                # if splitting succeeds
                if ns:
                    print "Start prediction..."
                    # dual training on validation pop set
                    for vdf in vdfs:
                        res = vdf.groupby(vdf.apply(
                            self.predict, axis=1)).apply(agg_by_attr)
                        # if not all validation sets match the same rank order, break, need to change a feature
                        if not self.is_same_order(res):
                            is_need_change = True
                            print "Validation data not match"
                            break
                    else:
                        # if all validation sets match, succeed, add the child nodes into next level
                        is_need_change = False
                        nodes.extend(ns)
                        print "All validation data satisfied!"
                    # change the feature
                    if is_need_change:
                        if len(node.attr_list) > 1:
                         # Update the selected feature of the node
                            for ind in xrange(1, len(node.attr_list)):
                                node.col = node.attr_list.index[ind]

                                if node.tb:
                                    node.tb.F1 = node.attr_list.iloc[ind]['F11']
                                if node.fb:
                                    node.fb.F1 = node.attr_list.iloc[ind]['F10']

                                # estimate the metric: F1
                                for vdf in vdfs:
                                    res = vdf.groupby(vdf.apply(
                                        self.predict, axis=1)).apply(agg_by_attr)
                                    if not self.is_same_order(res):
                                        print "Validation data not match"
                                        break
                                else:
                                    print "All validation data satisfied!"
                                    nodes.extend(ns)
                                    break
                            else:
                                # all feature looped but no qualified fearure found so as to meet the dual training
                                node.update_node(col=-1, tb=None, fb=None)
                                print 'All alternative looped at %d, found no suitable feature, stop splitting' % node.id_

                        else:
                            # only one feature
                            node.update_node(col=-1, tb=None, fb=None)
                            print 'No alternative feature at %d, stop splitting' % node.id_
                else:
                    # no child nodes generated after splitting
                    node.update_node(col=-1, tb=None, fb=None)
                    print 'Cannot split at %d, stop splitting' % node.id_
        print "Done! Tree-building finished"

    def predict(self, df):
        """predict a record assigned to which node"""
        this = self.root
        while this.col != -1:
            if df[this.col] == 1:
                this = this.tb
            elif df[this.col] == 0:
                this = this.fb
        return this.id_

    def is_same_order(self, res):
        """judge whether train and validation data has same rank order"""
        this = self.root

        F2_list = [search_Metric(this, id_) for id_ in res.sort_values(
            by='Metric', ascending=True).index]
        return F2_list == sorted(F2_list)

    def display(self):
        """print the tree"""
        this = self.root
        print "%4s %5s %8s %20s %10s %10s" % ("Node", "F1", "Size", "Attr", "Leftnode", "Rightnode")

        print "%4d %5.3f %8d %20s %10d %10d" % (this.id_, this.F1, this.size, this.col, this.tb.id_, this.fb.id_)
        this.tb.print_node()
        this.fb.print_node()

