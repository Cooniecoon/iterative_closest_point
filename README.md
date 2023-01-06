


### Example
```
    iterative_closest_point::ICP icp;
    icp.setSourceData(source_cloud);
    icp.setTargetData(target_cloud);
    icp.setEuclidianThreshold(0.08);
    icp.setMaxIteration(100);

    icp.setInitialTransformation(xyz, rpy);
    icp.computeTransformation();
    Eigen::Matrix4d tf = icp.getFinalTransformation();
```