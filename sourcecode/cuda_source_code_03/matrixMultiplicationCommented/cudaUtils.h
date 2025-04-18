//=============================================================================================
// Name        		: cudaUtils.h
// Author      		: Jose Refojo
// Version     		:	06-02-2018
// Creation date	:	06-02-2018
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This source code contains common utils for cuda programs
//=============================================================================================


int		chooseCudaCard				(bool verbose);				// How to pick the best available card
void	cudaLastErrorCheck			(const char *message);		// How to check the latest GPU error message
void	cudaTestBlockInformation	(dim3 myDimBlock);			// Test that this block size (number of threads per block) is valid
void	cudaTestGridInformation		(dim3 myDimGrid);			// Test that the number of blocks is valid
