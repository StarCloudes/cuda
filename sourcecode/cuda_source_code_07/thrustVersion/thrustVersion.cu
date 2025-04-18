//=============================================================================================
// Name        		: thrustVersion.cu
// Author      		: Jose Refojo
// Version     		:	25-02-2013
// Creation date	:	25-02-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will request and print the current thrust version
//=============================================================================================

#include <thrust/version.h>
#include <iostream>

int main(void)
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;

    std::cout << "Thrust v" << major << "." << minor << std::endl;

    return 0;
}
