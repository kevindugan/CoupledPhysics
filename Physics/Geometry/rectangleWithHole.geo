SetFactory("OpenCASCADE");

//mesh_size = 25.0;  // Fine
mesh_size = 100.0; // Coarse

Rectangle(1) = {0, 0, 0, 400.0, 600.0};
Disk(2) = {200.0, 300.0, 0, 50.0};

BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
MeshSize{ PointsOf{ Surface{1}; } } = mesh_size;
Recombine Surface{1};

//Physical Curve("yneg") = {1};
//Physical Curve("xneg") = {2};
//Physical Curve("xpos") = {3};
//Physical Curve("ypos") = {4};
//Physical Curve("inner") = {5};
//Physical Surface("region") = {1};

Mesh 2;