import numpy as np

def adjust_length(a,b,choose=[]): # add strt,end?
    '''a,b are diff length, make same length.
        choose = a, to make b same length as a'''
    la=len(a)
    lb=len(b)
    if la>lb:
        if choose is None or choose is a:
            b = np.pad(b,[0,la-lb])
        elif np.array_equal(choose, b):
            a = a[:lb]
    elif la<lb:
        if choose is None or choose is b:
            a = np.pad(a,[0,lb-la])
        elif np.array_equal(choose, a):
            b = b[:la]

    return a,b


def closest(lst, K):
     lst = np.asarray(lst) # convert to array
     indx = (np.abs(lst - K)).argmin() # find minimal absolute distance
     return indx

def assign_equal_width_bins(values, nbins, vmin=None, vmax=None):
    values = np.asarray(values)
    mask = ~np.isnan(values)
    if vmin is None:
        vmin = np.nanmin(values[mask])
    if vmax is None:
        vmax = np.nanmax(values[mask])

    bins = np.linspace(vmin, vmax, nbins + 1)
    groups = np.digitize(values, bins) - 1
    groups[groups < 0] = -1
    groups[groups >= nbins] = nbins - 1
    return groups, bins


def assign_percentile_bins(values, nbins):
    values = np.asarray(values)
    mask = ~np.isnan(values)
    percentiles = np.percentile(values[mask], np.linspace(0, 100, nbins + 1))
    groups = np.digitize(values, percentiles) - 1
    groups[groups < 0] = -1
    groups[groups >= nbins] = nbins - 1
    return groups, percentiles

def merge_groups_N_bins(groups=np.array([]), bins=np.array([]), merge_map=None):
    """ Merge groups according to merge_map, which is a dict of new_label: [old_labels]
        e.g. merge_map = {0: [0,1], 1: [2,3,4]} will merge old groups 0 and 1 into new group 0, 
        and old groups 2,3,4 into new group 1.
        bins will be merged: take min/max of old bin edges for each new group.
        this merges bins that are adjacent in the original grouping, 
        so it assumes that old labels correspond to adjacent bins.
    """
    new_groups = np.copy(groups)
    
    # Create sorted list of new labels to maintain order
    sorted_new_labels = sorted(merge_map.keys())
    new_bin_edges = []
    
    for new_label in sorted_new_labels:
        old_labels = merge_map[new_label]
        
        # Relabel groups
        for old_label in old_labels:
            new_groups[groups == old_label] = new_label
        
        # Collect bin edges for this merged group
        old_bin_edges = []
        for old_label in old_labels:
            if old_label < len(bins) - 1:
                old_bin_edges.extend([bins[old_label], bins[old_label + 1]])
        
        if old_bin_edges:
            # Add lower edge for first group, or just upper edge for subsequent
            if len(new_bin_edges) == 0:
                new_bin_edges.append(np.min(old_bin_edges))
            new_bin_edges.append(np.max(old_bin_edges))
    
    new_bins = np.array(new_bin_edges)
    return new_groups, new_bins
