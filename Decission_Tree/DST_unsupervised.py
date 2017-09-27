import numpy as np
import pandas as pd
import time
import gc
import json


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
        self.attrs = attrs    # alternative feature list
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

        # filter qualified info
        info = info[(info[["C2", "C3", "C4", "C6", "C7"]] == 1).all(axis=1)]
        info.sort_values(by="F2", ascending=0, inplace=True)
        # print info.iloc[:10,:]

        # if no feature retained return None
        if info.empty:
            self.update_node(size=df_size, set_=None)
            print '[Empty] No feature qualified'
            return None

        # choose top 1 attribute
        top_attr = info.iloc[0]

        # if F2>0 do regular split
        if top_attr.F2 > 0:
            print "Node %d successful" % self.id_
            V = top_attr.name

            left_node = decisionnode(F1=top_attr.F11, size=len(
                df[df[V] == 1]), id_=self.id_ * 2, p=self, set_=df[df[V] == 1])
            right_node = decisionnode(F1=top_attr.F10, size=len(
                df[df[V] == 0]), id_=self.id_ * 2 + 1, p=self, set_=df[df[V] == 0])
            self.update_node(col=V, size=df_size, F2=top_attr.F2,
                             tb=left_node, fb=right_node, attrs=info, set_=None)
            gc.collect()
            print self.tb.F1, self.fb.F1
            return [left_node, right_node]
        else:
            # `if F2==0 process` is identical with `F2>0` while the result probably makes no
            # business sense
            print "F2=0 at %d, split node randomly" % self.id_
            V = top_attr.name

            left_node = decisionnode(F1=top_attr.F11, size=len(
                df[df[V] == 1]), id_=self.id_ * 2, p=self, set_=df[df[V] == 1])
            right_node = decisionnode(F1=top_attr.F10, size=len(
                df[df[V] == 0]), id_=self.id_ * 2 + 1, p=self, set_=df[df[V] == 0])
            self.update_node(col=V, size=df_size, F2=top_attr.F2,
                             tb=left_node, fb=right_node, attrs=info, set_=None)
            gc.collect()
            print self.tb.F1, self.fb.F1
            return [left_node, right_node]

    def update_node(self, **kwargs):
        """for updating node while splitting"""
        attr_def =  set(["col", "id_", "tb", "fb", "size", "F1", "F2", "attrs", "set_"])
        for attr in kwargs:
            if attr in attr_def:
                self.__setattr__(attr, kwargs[attr])

    def print_node(self):
        """visualization"""
        if self.col != -1:
            print "%4d %5.3f %8d %20s %10d %10d" % (self.id_, self.F1, self.size, self.col, self.tb.id_, self.fb.id_)
            self.tb.print_node()
            self.fb.print_node()
        else:
            print "%4d %5.3f %8d %20s %10s %10s" % (self.id_, self.F1, self.size, self.col, "Null", "Null")


def clean_data(DT_df, attributes):
    """data preprocessing"""
    # DT_df = DT_df.drop(drop_cols, axis=1)
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
    result["TCM_MD"] = df[df.ctl_grp_ind == 0]["tcm_redeem_md"].sum()
    df = df.groupby(["ctl_grp_ind"]).agg({"xtra_card_nbr": "count", "fs_scan_amt_pre": "sum", "fs_scan_amt_pos": "sum",
                                          "fs_scan_amt_pos_PF": "sum", "dyn_margin_amt_pre": "sum", "dyn_margin_amt_pos": "sum",
                                          "dyn_margin_amt_pos_PF": "sum"})
    if df.shape != (2, 7):
        return None

    result["cnt_card_T"] = df.loc[0]["xtra_card_nbr"]
    result["cnt_card_C"] = df.loc[1]["xtra_card_nbr"]
    result["fs_scan_amt_pre_T"] = df.loc[0]["fs_scan_amt_pre"]
    result["fs_scan_amt_pre_C"] = df.loc[1]["fs_scan_amt_pre"]
    result["fs_scan_amt_pos_T"] = df.loc[0]["fs_scan_amt_pos"]
    result["fs_scan_amt_pos_C"] = df.loc[1]["fs_scan_amt_pos"]
    result["fs_scan_amt_pos_PF_T"] = df.loc[0]["fs_scan_amt_pos_PF"]
    result["fs_scan_amt_pos_PF_C"] = df.loc[1]["fs_scan_amt_pos_PF"]
    result["dyn_margin_amt_pre_T"] = df.loc[0]["dyn_margin_amt_pre"]
    result["dyn_margin_amt_pre_C"] = df.loc[1]["dyn_margin_amt_pre"]
    result["dyn_margin_amt_pos_T"] = df.loc[0]["dyn_margin_amt_pos"]
    result["dyn_margin_amt_pos_C"] = df.loc[1]["dyn_margin_amt_pos"]
    result["dyn_margin_amt_pos_PF_T"] = df.loc[0]["dyn_margin_amt_pos_PF"]
    result["dyn_margin_amt_pos_PF_C"] = df.loc[1]["dyn_margin_amt_pos_PF"]
    # result["cnt_redeemer"]=df[np.logical_and(df.ctl_grp_ind==0,df.ind_tcm_redeem==1)].count()

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
    result["cnt_card_all"] = result.cnt_card_T + result.cnt_card_C
    result["cnt_card_C"] = result.cnt_card_C
    # main metric for node splitting
    result["Metric"] = (result.NMI_PF + result.DYN_PF) / 2
    return result[["GSL", "NMI", "NMI_PF", "DYN", "DYN_PF", "PRE_VARIANCE", "cnt_card_all", "cnt_card_C", "Metric"]]


def check_search(df, var):
    """aggregate data by a certain feature"""
    info = pd.Series()
    df = df.groupby(var).apply(agg_by_attr)
  #     print g
    if not (df.shape[0] == 2 and isinstance(df.iloc[0], pd.Series)):
        #         print "Bad aggregation"
        return info
    # C1: GSL increase
    info["C2"] = 1 if all(abs(df.PRE_VARIANCE < 0.05)) else 0
    info["C3"] = 1 if all(1.1 * df.NMI > df.NMI_PF) else 0  #
    # C5: NMI_PF>0 and DYN_PF>0
    info["C4"] = 1 if all(1.1 * df.DYN > df.DYN_PF) else 0
    info["C6"] = 1 if all(df.cnt_card_all > 40000) else 0 #POP SIZE
    info["C7"] = 1 if all(df.cnt_card_C > 5000) else 0
    info["F11"] = df.loc[1].Metric
    info["F10"] = df.loc[0].Metric
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

    def dual_train(self, df, vdfs=[]):
        attributes = self.attributes
        # make a root node for a tree
        if not self.root:
            self.root = decisionnode(id_=1, set_=df)
            self.root.F1 = agg_by_attr(df).Metric
            nodes = self.root.split(attributes)
            # return
        # bfs algorithm, do while there is any node at this level
        while nodes:
            this_level = nodes
            nodes = []
            print "Node at this level: %r" % this_level
            # loop in all nodes at this level
            for node in this_level:
                # split this node
                ns = node.split(attributes)
                # if splitting succeeds
                if ns:
                    print "Start prediction..."
                    # dual training on validation pop set
                    for vdf in vdfs:
                        pv = vdf.apply(self.predict, axis=1).values
                        res = vdf.groupby(pv).apply(agg_by_attr)
                        # if not all validation sets match the same rank order,
                        # break, need to change a feature
                        if not self.is_same_order(res):
                            is_need_change = True
                            print "Validation data not match for feature %s" % node.col
                            break
                    else:
                        # if all validation sets match, succeed, add the child
                        # nodes into next level
                        is_need_change = False
                        nodes.extend(ns)
                        print "All validation data satisfied! Feature %s selected!" % node.col
                    # change the feature
                    if is_need_change:
                        if len(node.attrs) > 1:
                         # Update the selected feature of the node
                            for ind in xrange(1, len(node.attrs)):
                                node.col = node.attrs.index[ind]

                                if node.tb:
                                    node.tb.F1 = node.attrs.iloc[
                                        ind]['F11']
                                if node.fb:
                                    node.fb.F1 = node.attrs.iloc[
                                        ind]['F10']

                                # estimate the metric: F1
                                for vdf in vdfs:
                                    pv = vdf.apply(self.predict, axis=1).values
                                    res = vdf.groupby(pv).apply(agg_by_attr)
                                    if not self.is_same_order(res):
                                        print "Validation data not match for feature %s" % node.col
                                        gc.collect()
                                        break
                                else:
                                    print "All validation data satisfied! Feature %s selected!" % node.col
                                    nodes.extend(ns)
                                    break

                            else:
                                # all feature looped but no qualified fearure
                                # found so as to meet the dual training
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
        print("%4s %5s %8s %20s %10s %10s" % ("Node", "F1", "Size", "Attr", "Leftnode", "Rightnode"))
        if not this.tb or not this.fb:
            print("%4d %5.3f %8d %20s %10s %10s" % (this.id_, this.F1, this.size, this.col, "Null", "Null"))
        else:
            print("%4d %5.3f %8d %20s %10d %10d" % (this.id_, this.F1, this.size, this.col, this.tb.id_, this.fb.id_))
            this.tb.print_node()
            this.fb.print_node()


def clean_set(this):
    if this.col != -1:
        this.set_ = None
        this.attrs = None
        this.p = None
        clean_set(this.tb)
        clean_set(this.fb)
    else:
        this.set_ = None
        this.attrs = None
        this.p = None


def save_obj(obj, path):
    if isinstance(obj, myTree):
        root = obj.root
        root.set_ = None

        clean_set(root)
        with open(path, 'w+') as f:
            json.dump(root, f, default=lambda o: o.__dict__, indent=4, check_circular=False)


def main():

    cols = set(["xtra_card_nbr", "ctl_grp_ind", "mailer_version_id", "fs_scan_amt_pre", "fs_scan_amt_pos", "fs_scan_amt_pos_PF", "tcm_redeem_md", "dyn_margin_amt_pre", "dyn_margin_amt_pos", "dyn_margin_amt_pos_PF", "V_STORE_CLUSTER", "V_CVS_DISTANCE", "V_ENGAGE_PERS", "V_ENGAGE_unengae", "V_NEGATIVE_MARGIN_RATE_L12M", "V_NEGATIVE_MARGIN_RATE_L3M", "V_NEGATIVE_MARGIN_RATE_L6M", "V_MARGIN_RATE_TREND", "V_GEO_CLUSTER_P25", "V_CATEGORY_BREADTH_L3M_P25", "V_CATEGORY_BREADTH_P25", "V_CATEGORY_BREADTH_L6M_P25", "V_CAT_CNT_P25", "V_RATIO_CHRONIC_SPEND_P25", "V_RATIO_CHRONIC_SPEND_L6M_P25", "V_CPN_BS_L6M_P25", "V_RATIO_CHRONIC_SPEND_L3M_P25", "V_CPN_BS_L3M_P25", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_RATIO_P25", "V_FS_TRIP_NON_CPN_NON_PROMO_RATIO_P25", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_P25", "V_FS_TRIP_NON_CPN_NON_PROMO_P25", "V_REGULAR_BS_L6M_P25", "V_FS_SCAN_QTY_NON_CPN_RATIO_P25", "V_RATIO_FS_TRIP_WITH_CPN_P25", "V_FS_TRIP_WITH_CPN_P25", "V_FS_COST_AMT_NON_CPN_P25", "V_NON_CPN_BS_L12M_P25", "V_FS_SCAN_QTY_NON_CPN_P25", "V_BS_LIFT_L12M_P25", "V_NON_CPN_BS_L6M_P25", "V_NON_CPN_BS_L3M_P25", "V_tenure_P25", "V_MARGIN_RATE_L12M_P25", "V_CPN_DEPTH_NEW_P25", "V_PROMO_DEPTH_L12M_P25", "V_FS_AMT_WITH_CPN_P25", "V_CPN_BS_L12M_P25", "V_FS_TRIP_WT_PROMO_WT_CPN_P25", "V_FS_TRIP_WITH_PROMO_P25", "V_CPN_DEPTH_L12M_P25", "V_coupon_sales_ratio_P25", "V_FS_RETL_AMT_P25", "V_FS_SCAN_AMT_P25", "V_REGULAR_BS_L12M_P25", "V_RATIO_PROMO_TRIPS_P25", "V_FS_SCAN_QTY_P25", "V_RX_TRIP_P25", "V_FS_TRIP_P25", "V_REGULAR_BS_L3M_P25", "V_MARGIN_RATE_L3M_P25", "V_MARGIN_RATE_L6M_P25", "V_GEO_CLUSTER_P50", "V_CATEGORY_BREADTH_L3M_P50", "V_CATEGORY_BREADTH_P50", "V_CATEGORY_BREADTH_L6M_P50", "V_CAT_CNT_P50", "V_RATIO_CHRONIC_SPEND_P50", "V_RATIO_CHRONIC_SPEND_L6M_P50", "V_CPN_BS_L6M_P50", "V_RATIO_CHRONIC_SPEND_L3M_P50", "V_CPN_BS_L3M_P50", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_RATIO_P50", "V_FS_TRIP_NON_CPN_NON_PROMO_RATIO_P50", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_P50", "V_FS_TRIP_NON_CPN_NON_PROMO_P50", "V_REGULAR_BS_L6M_P50", "V_FS_SCAN_QTY_NON_CPN_RATIO_P50", "V_RATIO_FS_TRIP_WITH_CPN_P50", "V_FS_TRIP_WITH_CPN_P50", "V_FS_COST_AMT_NON_CPN_P50", "V_NON_CPN_BS_L12M_P50",
                "V_FS_SCAN_QTY_NON_CPN_P50", "V_BS_LIFT_L12M_P50", "V_NON_CPN_BS_L6M_P50", "V_NON_CPN_BS_L3M_P50", "V_tenure_P50", "V_MARGIN_RATE_L12M_P50", "V_CPN_DEPTH_NEW_P50", "V_PROMO_DEPTH_L12M_P50", "V_FS_AMT_WITH_CPN_P50", "V_CPN_BS_L12M_P50", "V_FS_TRIP_WT_PROMO_WT_CPN_P50", "V_FS_TRIP_WITH_PROMO_P50", "V_CPN_DEPTH_L12M_P50", "V_coupon_sales_ratio_P50", "V_FS_RETL_AMT_P50", "V_FS_SCAN_AMT_P50", "V_REGULAR_BS_L12M_P50", "V_RATIO_PROMO_TRIPS_P50", "V_FS_SCAN_QTY_P50", "V_RX_TRIP_P50", "V_FS_TRIP_P50", "V_REGULAR_BS_L3M_P50", "V_MARGIN_RATE_L3M_P50", "V_MARGIN_RATE_L6M_P50", "V_GEO_CLUSTER_P75", "V_CATEGORY_BREADTH_L3M_P75", "V_CATEGORY_BREADTH_P75", "V_CATEGORY_BREADTH_L6M_P75", "V_CAT_CNT_P75", "V_RATIO_CHRONIC_SPEND_P75", "V_RATIO_CHRONIC_SPEND_L6M_P75", "V_CPN_BS_L6M_P75", "V_RATIO_CHRONIC_SPEND_L3M_P75", "V_CPN_BS_L3M_P75", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_RATIO_P75", "V_FS_TRIP_NON_CPN_NON_PROMO_RATIO_P75", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_P75", "V_FS_TRIP_NON_CPN_NON_PROMO_P75", "V_REGULAR_BS_L6M_P75", "V_FS_SCAN_QTY_NON_CPN_RATIO_P75", "V_RATIO_FS_TRIP_WITH_CPN_P75", "V_FS_TRIP_WITH_CPN_P75", "V_FS_COST_AMT_NON_CPN_P75", "V_NON_CPN_BS_L12M_P75", "V_FS_SCAN_QTY_NON_CPN_P75", "V_BS_LIFT_L12M_P75", "V_NON_CPN_BS_L6M_P75", "V_NON_CPN_BS_L3M_P75", "V_tenure_P75", "V_MARGIN_RATE_L12M_P75", "V_CPN_DEPTH_NEW_P75", "V_PROMO_DEPTH_L12M_P75", "V_FS_AMT_WITH_CPN_P75", "V_CPN_BS_L12M_P75", "V_FS_TRIP_WT_PROMO_WT_CPN_P75", "V_FS_TRIP_WITH_PROMO_P75", "V_CPN_DEPTH_L12M_P75", "V_coupon_sales_ratio_P75", "V_FS_RETL_AMT_P75", "V_FS_SCAN_AMT_P75", "V_REGULAR_BS_L12M_P75", "V_RATIO_PROMO_TRIPS_P75", "V_FS_SCAN_QTY_P75", "V_RX_TRIP_P75", "V_FS_TRIP_P75", "V_REGULAR_BS_L3M_P75", "V_MARGIN_RATE_L3M_P75", "V_MARGIN_RATE_L6M_P75", "V_MD_RED_PERS_P25",  "V_MD_RED_MASS_P25", "V_MD_RED_PERS_P50", "V_MD_RED_BASE_P50", "V_MD_RED_MASS_P50", "V_MD_RED_BASE_P75", "V_MD_RED_MASS_P75", "V_HH_LEVEL_ELIGIBLE_IND", "V_HH_NonTCM_BIN_1", "V_HH_NonTCM_BIN_2", "V_HH_NonTCM_BIN_3", "V_HH_NonTCM_BIN_4", "V_HH_NonTCM_BIN_5", "V_HH_NonTCM_BIN_6", "V_HH_NonTCM_BIN_7", "V_HH_NonTCM_BIN_8", "V_HH_NonTCM_BIN_9", "V_HH_LEVEL_pred_tgt_NonTCM_P25", "V_HH_LEVEL_pred_tgt_NonTCM_P50", "V_HH_LEVEL_pred_tgt_NonTCM_P75"])
    drop_cols = set(["V_ENGAGE_PERS", "V_IND_BUSINESS_RULE_TAG", "V_NEGATIVE_MARGIN_RATE_L3M", "V_NEGATIVE_MARGIN_RATE_L6M",
                     "V_MARGIN_RATE_TREND", "V_RATIO_CHRONIC_SPEND_P25", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_P25", "V_FS_TRIP_NON_CPN_NON_PROMO_P25",
                     "V_TCM_RED_RATIO_DYN_MARGIN_L6_L12_P25", "V_FS_SCAN_QTY_NON_CPN_P25", "V_RATIO_FS_TRIP_TCM_RED_L3M_P25", "V_FS_AMT_WITH_CPN_P25",
                     "V_CPN_BS_L12M_P25", "V_REGULAR_BS_L12M_P25", "V_RATIO_PROMO_TRIPS_P25", "V_FS_TRIP_P25", "V_REGULAR_BS_L3M_P25",
                     "V_MARGIN_RATE_L3M_P25", "V_CONV_SPEND_P25", "V_CONV_SPEND_L6M_P25", "V_COUPON_SALES_RATIO_TCM_RED_L3M_P50",
                     "V_RATIO_FS_TRIP_TCM_RED_L3M_P50", "V_CATEGORY_BREADTH_L6M_P75", "V_FS_SCAN_QTY_NON_CPN_NON_PROMO_RATIO_P75",
                     "V_REGULAR_BS_L6M_P75", "V_CPN_DEPTH_NEW_P75", "V_MD_RED_PERS_P75", "V_HH_TCM_BIN_3", "V_HH_TCM_BIN_4",
                     "V_HH_LEVEL_pred_tgt_NonTCM_P25", "V_HH_LEVEL_pred_tgt_TCM_P25", "V_HH_LEVEL_pred_tgt_TCM_P50",
                     "V_HH_TCM_SCORE", "V_HH_NONTCM_SCORE", "V_GEO_CLUSTER_P25", "V_GEO_CLUSTER_P50", "V_GEO_CLUSTER_P75",
                     "V_WTS_CLUSTER","V_MD_RED_BASE_P25"])
    cols = cols.difference(drop_cols)
    print 'Reading Train data'
    # train data
    DT_file1 = '/home/shaoze.luo2/data/tcm_decision_tree_20170810/combine_all_all_p1r.csv'
    DT_file2 = '/home/shaoze.luo2/data/tcm_decision_tree_20170810/combine_all_all_p2r.csv'
    DT_df1 = pd.read_csv(DT_file1, sep="|", header=0, usecols=cols).fillna(0)
    DT_df2 = pd.read_csv(DT_file2, sep="|", header=0, usecols=cols).fillna(0)
    print 'Reading Validation data'
    # validation data
    val_file1 = '/home/shaoze.luo2/data/tcm_decision_tree_20170810/pull_dual_set_6122.csv'
    val_df1 = pd.read_csv(val_file1, sep="|", header=0, usecols=cols).fillna(0)
    val_file2 = '/home/shaoze.luo2/data/tcm_decision_tree_20170810/pull_dual_set_6512.csv'
    val_df2 = pd.read_csv(val_file2, sep="|", header=0, usecols=cols).fillna(0)

    # attributes to be tested
    attributes = [a for a in DT_df1.columns if a.startswith(
        'V') and a != "V_CNT_RECV" and a != "V_CNT_RED"]

    constraint = ["C2", "C3", "C4", "C6", "C7"]
    random_state = 128
    print 'Cleaning data'
    DT_df1 = clean_data(DT_df1, attributes)  # .sample(frac=1, random_state=21)
    DT_df2 = clean_data(DT_df2, attributes)  # .sample(frac=1, random_state=21)
    DT_df = pd.concat([DT_df1, DT_df2], ignore_index=True)
    val_df1 = val_df1.sample(frac=0.2, random_state=random_state)
    # .sample(frac=1, random_state=21)
    val_df1 = clean_data(val_df1, attributes)
    val_df2 = val_df2.sample(frac=0.2, random_state=random_state)
    # .sample(frac=1, random_state=21)
    val_df2 = clean_data(val_df2, attributes)
    print "Data preprocess done!"
    print "Tree building"
    # bulid a tree
    tree = myTree(attributes=attributes)
    gc.enable()
    tree.dual_train(DT_df, [val_df1, val_df2])
    # display
    tree.display()
    save_obj(tree, 'tree_20170810_w_dual.dat')

if __name__ == '__main__':
    main()
