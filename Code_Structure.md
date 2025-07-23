
```python
def processSegment()

for each seg
    computeSegmentLabelCandidates()

    for each point
        getBlockPtr() & getVoxelPtr()
            Find the corres. voxel given the current point position
        getNextUnassignedLabel()
            Get the glo-label of the current voxel, if it has not beed assigned to other segments,
            Otherwise, find unassigned other candidate labels with highest conf in the voxel
        increaseLabelCountForSegment()
            check if the given label has been listed in the candidate list
            - if in, increase the overlap count for this pair of <label, segment>
            checkForSegmentLabelMergeCandidate() 
                check if the overlapping is enough, add this pair to merge list if enough
            - if not (this glo-label is unseen), then add it to the candidate list

    obtain all the possible glo-label candidates from the cache

    if none glo-label observed, then find this segment a new glo-label

def integrateFrame()

decideLabelPointClouds()
    while -> getNextSegmentLabelPair()
        iterate all labels, find the segment with max overlapping count
        pair these two <max_label, max_seg>, assign this label if unassigned to other segment

        for all seen segments fulfill the conditions in the searching
            first remove it from all labels from label_to_seg_cands

            computeSegmentLabelCandidates() recompute the candidate labels for this segment
        
        return the pair

        insert the pair to candidate pairs list
    
    for all segments
        increasePairwiseConfidenceCount(), increase the label-to-label conf. for all label candidates under each segment

    set new label for all remaining segments without a assigned glo-label

    # ====== instance segmentation in Voxblox++=====
    for each new segment
        check if the local-inst-id has mapped to a glo-inst-id

        - if yes, increase the label_to_inst count
        - if no, getInstanceLabel()
            find the glo-inst linked to this segment with highest count
            - if inst-id ==0, create a new glo-inst
            - else increase the label_to_inst count with the found glo-inst-id
                NOTE here we also want to make sure we only pair one segment to each glo-inst

        increase the class count of this label using the 2D semantic

    # ====== instance segmentation in VSC (Yang et. al.)=====
    for each new segment
        find all other segment with unassigned label and with same instance id and semantic id

        calculate the inner, external confidence for all seleceted segs and put into a SegGraph

        insertInstanceToSegGraph()
            

insert()
    for each seg integratePointCloud() 
        bundleRays(), integrate all points projected into the same voxel
        integrateRays(), update TSDF
            integrateVoxels(), average all points and do ray casting
            updateLayerWithStoredBlocks()
            updateLabelLayerWithStoredBlocks()
        integrateRays with clear_ray set to true
```