candidate no.:201332107

cornell box colours applied as directed
parameters for specular, diffuse, ambient chosen on best looks
static flags in code to turn off direct lighting and indirect lighting


parameters and scene setup

//cornell box
triangle wallleft1(1,  vec3(1, 1, 2), vec3(1, -1, 1), vec3(1, 1, 1), Material{vec3(255, 0, 0), 0.1, 0.1, 0.1}, false);
triangle wallleft2(2,  vec3(1, -1, 2), vec3(1, -1, 1), vec3(1, 1, 2), Material{vec3(255, 0, 0), 0.1, 0.1, 0.1}, false);
triangle wallright1(3, vec3(-1, -1, 1), vec3(-1, 1, 2), vec3(-1, 1, 1), Material{vec3(0, 255, 0), 0.1, 0.1, 0.1}, false);
triangle wallright2(4, vec3(-1, -1, 1), vec3(-1, -1, 2), vec3(-1, 1, 2), Material{vec3(0, 255, 0), 0.1, 0.1, 0.1}, false);
triangle wallback1(5,  vec3(-1, 1, 2), vec3(-1, -1, 2), vec3(1, -1, 2), Material{vec3(255, 255, 255), 0.1, 0.1, 0.1}, false);
triangle wallback2(6,  vec3(-1, 1, 2), vec3(1, -1, 2), vec3(1, 1, 2), Material{vec3(255, 255, 255), 0.1, 0.1, 0.1}, false);

triangle floor1(7, vec3(1, -1, 1), vec3(1, -1, 2), vec3(-1, -1, 2), Material{vec3(255,255,255), 0.1, 0.1, 0.1}, false);
triangle floor2(8, vec3(-1, -1, 2), vec3(-1, -1, 1), vec3(1, -1, 1), Material{vec3(255,255,255), 0.1, 0.1, 0.1}, false);

//for use as area light set emission to true (last variable)
triangle roof1 = triangle(9,vec3(1, 1, 2), vec3(1, 1, 1), vec3(-1, 1, 2), Material{vec3(255,255,255), 0.1, 0.1, 0.1}, false);
triangle roof2 = triangle(10, vec3(-1, 1, 1), vec3(-1, 1, 2), vec3(1, 1, 1), Material{vec3(255,255,255), 0.1, 0.1, 0.1}, false);

//--area light setup
areaLight al(0.4, roof1, colour(255, 255, 255), 0.01, 0.01, 100, 0.1);
areaLight al2(0.4, roof2, colour(255, 255, 255), 0.01, 0.01, 100, 0.1);
// --area light setup

//uncomment for area lights
//w.areaLights.push_back(al); //--area lights that are not working correctly
//w.areaLights.push_back(al2);

//--point light setup
vec3 lightLocation = vec3{0, 0, -1};
pointLight l = pointLight{colour(255, 255, 255), lightLocation, 0.01, 0.1, 80.0, 0.01};
w.pointLights.push_back(l); //uncomment for point light
//--point light setup

variables are chosen based on visibility, point lights and area lights without tweaking of values will cause
the channels to overflow. lights are incredibly sensitive.
