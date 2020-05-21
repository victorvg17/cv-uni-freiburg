#include "CFilter.h"
#include "CMatrix.h"
#include <iostream>
#include<math.h>

using namespace std;
using namespace NFilter;

#define TAU 0.1

void non_linear_diffusion_filter(CMatrix<float>& boat_noise,
                                 CMatrix<float>& dx,
                                 CMatrix<float>& dy,
                                 CMatrix<float>& boat_output) {
  float w1, w2, w3, w4;
  float w5, w6, w7, w8;


  for (int y = 1; y < boat_output.ySize()-1; ++y) {
    for (int x = 1; x < boat_output.xSize()-1; ++x) {
      w1 = ( exp(-pow(dx(x, y), 2)) + exp(-pow(dx(x+1, y), 2)) )/2;
      w2 = ( exp(-pow(dx(x, y), 2)) + exp(-pow(dx(x-1, y), 2)) )/2;
      w3 = ( exp(-pow(dx(x, y), 2)) + exp(-pow(dx(x, y+1), 2)) )/2;
      w4 = ( exp(-pow(dx(x, y), 2)) + exp(-pow(dx(x, y-1), 2)) )/2;
      
      w5 = w1*boat_noise(x+1, y);
      w6 = w2*boat_noise(x-1, y);
      w7 = w3*boat_noise(x, y+1);
      w8 = w4*boat_noise(x, y-1);

      //get the output pixel value
      boat_output(x, y) = (1 - TAU*(w1 + w2 + w3 +w4))*boat_noise(x, y);
      boat_output(x, y) += TAU*(w5 + w6 + w7+ w8);
    }
  }
}

int main() {
  //load input image
  CMatrix<float> boat_noise;
  boat_noise.readFromPGM("BoatsNoise10.pgm");

  cout << "image size before mirroring x: " << boat_noise.xSize() << "y: " << boat_noise.ySize() << endl;

  //define a filter with size=3
  CDerivative<float> aDerivative(3);
  CMatrix<float> dx(boat_noise.xSize(),boat_noise.ySize());
  CMatrix<float> dy(boat_noise.xSize(),boat_noise.ySize());
  NFilter::filter(boat_noise,dx,aDerivative,1);
  NFilter::filter(boat_noise,dy, 1, aDerivative);

  // dx.writeToPGM("boat_dx.pgm");
  // dy.writeToPGM("boat_dy.pgm");

  cout << "dx shape " << dx.xSize() << ", " << dx.ySize() << endl;

  //initialise output image as a copy of boat_noise
  CMatrix<float> boat_output(boat_noise);

  non_linear_diffusion_filter(boat_noise, dx, dy, boat_output);
  
  //save output image
  boat_output.writeToPGM("boat_output.pgm");

}