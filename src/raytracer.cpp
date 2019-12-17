 #include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <random>

using namespace std;

/*
    vec3 to help with vector operations
*/
struct vec3
{
    float x;
    float y;
    float z;

    vec3(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    vec3()
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    float length()
    {
        return sqrtf(x * x + y * y + z * z);
    }

    void normalise()
    {
        x = x / length();
        y = y / length();
        z = z / length();
    }

    vec3 operator-(vec3 &other)
    {
        return vec3(this->x - other.x, this->y - other.y, this->z - other.z);
    }

    // vec3 operator*(float &scalar)
    // {
    //     return vec3(this->x * scalar, this->y * scalar, this->z * scalar);
    // }

    vec3 operator*(float scalar)
    {
        return vec3(this->x * scalar, this->y * scalar, this->z * scalar);
    }

    vec3 operator+(vec3 other)
    {
        return vec3(this->x + other.x, this->y + other.y, this->z + other.z);
    }

    vec3 operator/(float scalar)
    {
        // cout << "divided " << this->z << endl;
        return vec3(this->x / scalar, this->y / scalar, this->z / scalar);
    }

    vec3 operator+(float other)
    {
        return vec3(this->x + other, this->y + other, this->z + other);
    }
};

vec3 cross(vec3 &A, vec3 &B)
{
    float x = (A.y * B.z) - (A.z * B.y);
    float y = (A.z * B.x) - (A.x * B.z);
    float z = (A.x * B.y) - (A.y * B.x);
    float norm = sqrtf(pow(x, 2) + pow(y, 2) + pow(z, 2));
    return vec3(x, y, z);
};

float dot(vec3 &A, vec3 &B)
{
    return (A.x * B.x) + (A.y * B.y) + (A.z * B.z);
}

/*
    eye/camera which encodes the position, direction from which the scene is viewed, and its up and fov 
*/
struct eye
{
    vec3 position = vec3(0.0, 0.0, 0.0);
    vec3 direction = vec3(0.0, 0.0, 1.0);
    vec3 up = vec3(0.0, 1.0, 0.0);
    float fov = 90.0;

    eye(vec3 position, vec3 direction, vec3 up, float fov)
    {
        this->position = position;
        this->direction = direction;
        this->up = up;
        this->fov = fov;
    }
};

/*
    vec3 redefinition for colour data
*/
typedef vec3 colour;

/*
    helper function to blend two colours together by simply multiplying them
    @param a colour provided between the range 0-255 per channel
    @param b colout provided between the range 0-1 per channel
*/
colour blend(colour a, colour b){
    return colour(a.x * b.x, a.y * b.y,a.z * b.z);
}

/*
    point light which encodes its colour, position, diffuse intensity, specular intensity and ambient intensity as well as the specular coefficient
*/
struct light
{
    colour rgb;
    vec3 position;
    float diffuseIntensity;
    float specularIntensity;
    float specularCoeff;
    float ambientIntensity;
};

struct pointLight : light{};

//need to do some refactoring for light sources TODO
struct areaLight : light
{
    triangle t;

    surfel generateSamplePoint(){
        //P = (1 - sqrt(r1)) * A + (sqrt(r1) * (1 - r2)) * B + (sqrt(r1) * r2) * C
        float r1 = static_cast<float>(rand()) / static_cast<float>(1);
        float r2 = static_cast<float>(rand()) / static_cast<float>(1);

        vec3 point = (t.A * (1 - sqrtf64(r1))) + (t.B * (sqrtf64(r1) * (1 - r2))) + (t.C * (sqrtf64(r1)*r2));
        return surfel(t.m, point);

    }
};

/*
    Material which encodes the colour, diffuse intensity, specular intensity and ambient intensity
*/
struct Material
{
    colour rgb;
    float diffuseIntensity;
    float specularIntensity;
    float ambientIntensity;
    bool emission;

    bool emits(){
        return emission;
    }
};

struct surfel{
    Material m;
    vec3 point;
    vec3 normal;
    triangle t;
    bool emits;
    float extinctionProb;

    surfel();

    surfel(Material m, vec3 point){
        this->m = m;
        this->point = point;
        this->normal = normal;
    }

    vec3 getImpulseScatterDirection(vec3 direction){

    }
};

/*
    triangle which enodes its three vertices
*/
struct triangle
{
    vec3 A = vec3(61, 10, 1);
    vec3 B = vec3(100, 100, 1);
    vec3 C = vec3(25, 90, 1);
    bool emits = false;

    Material m;

    triangle(){}

    triangle(vec3 A, vec3 B, vec3 C, Material m)
    {
        this->A = A;
        this->B = B;
        this->C = C;
        this->m = m;
    }
};

/*
    ray which encodes the origin and direction
*/
struct ray
{
    vec3 start;
    vec3 direction;

    ray(vec3 start, vec3 direction)
    {
        this->start = start;
        this->direction = direction;
    }
};

/*
    holds all the triangles in the scene
*/
struct World{
    vector<triangle> tris;

    vector<areaLight> areaLights;
    vector<pointLight> pointLights;

    /*
        calculates if the given ray intersects a triangle and stores its surfel in an out param
    */
    bool intersect(ray r, surfel out){
        vec3 point;
        for(triangle t : tris){
            if(isIntersectingTriangle(r, t, point)){
                out.m = t.m;
                out.point = point;
                out.normal = getPlaneNormal(t);
                out.emits = t.emits;
                return true;
            }
        }
        return false;
    };

    void addTri(triangle t){
        tris.push_back(t);
    }
};

/*
    get the normal of a triangle and as a consequence the normal of the plane the triangle is on
    @param t the triangle for which the normal will calculated provided in world coordinates
    @return vec3 normal of the trianlge/plane in world coordinates
*/
vec3 getPlaneNormal(triangle t)
{
    vec3 planeDim1 = t.B - t.A;
    vec3 planeDim2 = t.C - t.A;
    vec3 normal = cross(planeDim1, planeDim2);
    normal.normalise();
    return normal;
}

/*
    check if a point is inside a triangle by checking if the point is on the left of every edge
    @param t the triangle to check if the point is inside provided in world coordinates
    @param point the point to check for provided in world coordintaes
    @param planeNormal the normal of the plane on which the triangle and point are on provided in world coordinates
    @return true if the point is inside the triangle, false otherwise
*/
bool isInsideTriangle(triangle t, vec3 point, vec3 planeNormal)
{
    vec3 triNormal1;
    vec3 triNormal2;
    vec3 triNormal3;

    vec3 edge0 = t.B - t.A;
    vec3 edge1 = t.C - t.B;
    vec3 edge2 = t.A - t.C;

    vec3 p0 = point - t.A;
    vec3 p1 = point - t.B;
    vec3 p2 = point - t.C;

    triNormal1 = cross(edge0, p0);
    triNormal2 = cross(edge1, p1);
    triNormal3 = cross(edge2, p2);

    float lTriNormal1 = triNormal1.length();
    float lTriNormal2 = triNormal2.length();
    float lTriNormal3 = triNormal3.length();

    float dot1 = dot(planeNormal, triNormal1);
    if (dot1 < 0)
        return false;
    float dot2 = dot(planeNormal, triNormal2);
    if (dot2 < 0)
        return false;
    float dot3 = dot(planeNormal, triNormal3);
    if (dot3 < 0)
        return false;

    return true;
}


/*
    check if a ray is intersecting a triangle
    @param r the ray to use for checking intersection provided in world coordinates
    @param t the triangle to check for intersection provided in world coordinates
    @param pointOut the point at which the ray intersects the triangle (only set if there is an intersection)
    @return true if there is an intersection and false otherwise
*/
bool isIntersectingTriangle(ray r, triangle t, vec3 &pointOut)
{
    vec3 normal = getPlaneNormal(t);

    float denom = dot(normal, r.direction);

    if (denom == 0)
    {
        return false;
    }

    vec3 numerator = t.A - r.start;
    float d = dot(numerator, normal) / denom;

    if (d < 0)
    {
        return false;
    }

    vec3 p = r.start + (r.direction * d);
    // cout << p.x << " " << p.y << " " << p.z << endl;
    if (isInsideTriangle(t, p, normal))
    {
        pointOut = p;

        return true;
    }
    return false;
}

/*
    calculate the distance of a point from an edge
    @param point the point to calculate the distance for
    @param vertex1 vertex at the end of the edge
    @param vertex2 vertex at the other end of the edge
    @return float the distance of the point from the edge
*/
float distance(vec3 point, vec3 vertex1, vec3 vertex2)
{
    //point in relation to one end of line
    float x = point.x - vertex1.x;
    float y = point.y - vertex1.y;
    //vector on line
    float ex = vertex2.x - vertex1.x;
    float ey = vertex2.y - vertex1.y;
    //normal to line
    float invex = -ey;
    float invey = ex;
    //dot product of point vector and normal
    float dotproduct = (x * invex) + (y * invey);
    //magnitude of normal
    float magnitudeNormal = sqrt((invex * invex) + (invey * invey));
    //removing scaling from distance
    return dotproduct / magnitudeNormal;
}

/*
    helper function to set image colour to light yellow (255, 255, 129)
    @param image the image buffer to setup
    @param width the width of the image
    @param height the height of the image
*/
void setupImage(vector<vector<int>> &image, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image[i][j * 3] = 255;
            image[i][j * 3 + 1] = 255;
            image[i][j * 3 + 2] = 192;
        }
    }
}

/*
    use barycentric interpolation to determine the colour of a point on a triangle
    @param point the point on the triangle in world coordinates
    @param t the triangle in world coordinates
    @return colour final colour of this point in range 0-255 for R, G, B
*/
colour baryinterp(vec3 point, triangle t)
{
    int R = 0;
    int G = 0;
    int B = 0;
    float alpha = distance(point, t.B, t.C) / distance(t.A, t.B, t.C); //distance from xstep,ystep to CB
    float beta = distance(point, t.A, t.C) / distance(t.B, t.A, t.C);  //distance from xstep,ystep to AC
    float gamma = distance(point, t.B, t.A) / distance(t.C, t.B, t.A); //distance from xstep,ystep to BA
    R = 255 * alpha;
    G = 255 * beta;
    B = 255 * gamma;
    vec3 colour(255,255,129);
    colour.x = (R < 0) ? 0 : (R > 255) ? 255 : R;
    colour.y = (G < 0) ? 0 : (G > 255) ? 255 : G;
    colour.z = (B < 0) ? 0 : (B > 255) ? 255 : B;
    return colour;
};

/*
    cast a ray from the origin to the location
    @param origin the point form which to cast provided in world coordinates
    @param the direction in which to cast provided in world coordinates
    @return ray which encodes the start position and computed direction
*/
ray castray(vec3 origin, vec3 location)
{
    vec3 direction = location - origin;
    //direction.normalise();
    return ray(origin, direction);
}

/*
    convert pixel coordinates into world coordinates (this will only work for square images and an fov of 90 degrees)
    @coord the pixel coordinates
    @width the width of the image 
    @height the height of the image
*/
vec3 convertCoordinates(vec3 coord, int width, int height)
{
    float aspectRatio = float(width) / float(height);
    float xr = ((2 * ((coord.x) / width)) - 1) * aspectRatio;
    float yr = ((2 * ((coord.y) / height)) - 1) * aspectRatio;
    return vec3(xr, yr, coord.z);
}

/*
    cast shadow ray to determine if point is in a shadow
    @param point the point from which to cast the shadow ray
    @normal the normal of the plane on whic the point lives
    @t the triangle to check shadow ray intersection against
    @l the light source to which the shadow ray is cast
    @return true if intersecting with triangle false otherwise
*/
bool isInShadow(vec3 point, vec3 normal, triangle t, light l)
{
    //triangle tWorld = triangle(convertCoordinates(t.A, 128, 128), convertCoordinates(t.B, 128, 128), convertCoordinates(t.C, 128, 128), t.m);
    ray r = castray(point + 0.01, l.position);
    vec3 pos;
    cout << r.direction.x << " " << r.direction.y << " " << r.direction.z << endl;
    if (isIntersectingTriangle(r, t, pos))
    {
        cout << "yes" << endl;
        return true;
    }
    return false;
}

/*
    write image buffer to ppm file
*/
void outputImage(ofstream &image, vector<vector<int>> &imageBuffer, int width, int height)
{
    image << "P3" << endl;
    image << "#" << endl;
    image << "128 128" << endl;
    image << "255" << endl;
    for (int ystep = height - 1; ystep >= 0; ystep--)
    {
        for (int xstep = 0; xstep < width; xstep++)
        {
            image << imageBuffer[ystep][xstep * 3] << " "
                  << imageBuffer[ystep][xstep * 3 + 1] << " "
                  << imageBuffer[ystep][xstep * 3 + 2] << " ";
        }
        image << endl;
    }
};

static World w;

//gamma correction = rgb ^ 1 / y where y = 2.2, repeat for each channel
//check this through TODO
colour gammaCorrection(colour c, float gamma){
    colour out;
    out.x = powf64(c.x, 1.0f/gamma);
    out.y = powf64(c.y, 1.0f/gamma);
    out.z = powf64(c.z, 1.0f/gamma);
    return out;
}

vec3 estimateDirectPointLight(surfel s, ray r, vector<pointLight>sources){
    vec3 out;
    for(pointLight l : sources){
        if(!isInShadow(s.point, s.normal, s.t, l)){
            vec3 omega = l.position - s.point;
            float dist = omega.length();
            vec3 irradiance = l.rgb / (4 * M_PI * dist * dist);
            //calculate output
        }
    }
    return out;
}

vec3 estimateDirectAreaLight(surfel s, ray r, vector<areaLight> sources){
    vec3 out;
    for(areaLight l : sources){
        surfel ls = l.generateSamplePoint();
        if(!isInShadow(s.point, ls.point, s.t, l)){
            vec3 omega = ls.point - s.point;
            float dist = omega.length();
            //calculate output
        }
    }
    return out;
}

vec3 randomBounceDir(vec3 normal){

}

vec3 estimateIndirectLight(surfel se, ray r, bool isEyeRay){
    if(rand() > se.extinctionProb){
        return vec3(0.0, 0.0, 0.0);
    }else{
        vec3 bounce = randomBounceDir(se.normal);
        ray bounceRay(se.point, bounce);
        return pathTrace(bounceRay, isEyeRay);
    }

}

vec3 pathTrace(ray r, bool isEyeRay){
    vec3 output(0.0,0.0,0.0);
    surfel se;
    if(w.intersect(r, se)){
        //we hit a area light source on first bounce
        if(isEyeRay && se.emits){
            output = output + se.m.rgb; //add emissive term to output
        }
        //if its not an eye ray or some other condition (WIP) estimate direct light
        if(!isEyeRay){
            //caculate the emitted area light source stuff here and point light source stuff here
            output = output + estimateDirectPointLight(se, r, w.pointLights);
            output = output + estimateDirectAreaLight(se, r, w.areaLights);
        }
        if(!(isEyeRay)){
        //calculate impulse scattering here and recurse
            output = output + estimateImpulseScattering(se, r, isEyeRay);
        }
        if(!isEyeRay){
            output = output + estimateIndirectLight(se, r, isEyeRay);
        }
    }
    return output;
};

vec3 estimateImpulseScattering(surfel se, ray r, bool isEyeRay){
    vec3 impulseDir = se.getImpulseScatterDirection(r.direction * -1.0f);
    ray newRay(se.point, impulseDir);
    return pathTrace(newRay, isEyeRay);
}

int main(int argc, char ** argv){
    
    int height = 128;
    int width = 128;

    vector<vector<int>> imageBuffer;
    setupImage(imageBuffer, width, height);


    triangle t(vec3(-0.04688, -0.84375, 1), vec3(0.5625, 0.5625, 1), vec3(-0.60938, 0.40625, 1), Material{colour(125, 125, 125), 0.1, 0.1, 0.01});
    vec3 lightLocation = vec3{0, 0, -1};
    light l = light{colour(0, 255, 129), lightLocation, 0.01, 0.1, 100.0, 0.1};
    eye e = eye(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), 90.0);

    for(int i = height-1; i >= 0; i--){
        for(int j = 0; j < width; j++){

        }
    }

    ofstream out("image.ppm");
    outputImage(out, imageBuffer, width, height);

}