# Literature reviews
    ## OpenGS-SLAM
        ### check how opengs-slam combine the results from YOLO-World and SAM
        ### 
    ## VSC 3D Pano Mapping
        ### Get panoptic seg from Mask2Former
        ### Get depth seg from one component of Voxblox++
        ### refine the seg from depth and pack all segs
        ### run the main mapping and merging pipeline
    ## Search for recent depth segmentation methods
        ### 
    ##

# Pipelines
    ## use SAM / something to do segmentation 
    ## get encoded features for each of the segmented area
        ### simple test - convert the NYU40 labels into one-hot encoding
        ### merge ?
        ### split ?


    ## Look-up table
        ### for each input 

    ### which method os better ? 
    # compare with the stored semantic feature in the current voxel and group them based on those limited options
    # compare with all existing semi-open-set labels in the look-up table and assign the corres sem-label (O())

# integrateFrame (gsm_py.cpp)
    # for each seg -> computeSegmentLabelCandidates ()
        # getBlockPtr & getVoxelPtr
        # getNextUnassignedLabel
        # increaseLabelConfidenceForSegment or increaseLabelCountForSegment
        # if not label got -> getFreshLabel
        # insert the new segment to the candidate map
    # decideLabelPointClouds
        # while -> getNextSegmentLabelPairWithConfidence
            # collect segments
        # for each seg_merge_cand -> increasePairwiseConfidenceCount
            # assign all unlabeled segs a new label

    # for each seg -> integratePointCloud 
        # bundleRays (integrate all points projected into the same voxel)
        # integrateRays (update TSDF)
            # integrateVoxels (multi-processing)
                # average all points and do ray casting
            # updateLayerWithStoredBlocks
            # updateLabelLayerWithStoredBlocks
        # integrateRays with clear_ray set to true

    # mergeLabel
        # updatePairwiseConfidenceAfter

    # getLabelsToPublish