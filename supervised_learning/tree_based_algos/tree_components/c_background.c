/*
c_background.c

implements more or less simple C functions that alter data passed in via a pointer. It is used for faster functions with Cython/Numpy.
*/


int c_smaller_greater(double* X, int* idx, int n, double splitVal) {
	int i;
	int lar = n-1;
	int sma = 0;
	for (i=0 ; i<n ; i++) {
		if(X[i] <= splitVal) { idx[sma] = i; sma++; }
		if(X[i] >  splitVal) { idx[lar] = i; lar--; }
	}
	return sma;
}

void c_copy_column(double* X, double* X_column, int nr_rows, int nr_cols, int col_idx) {
	int i;
	for(i=0; i<nr_rows; i++) X_column[i] = X[nr_cols * i + col_idx];
	return;
} 

void c_create_X_children(double* X, int* Y, double* X_left, double* X_right, int* Y_left, int* Y_right, int* idx_array, int split_idx, int nr_rows, int nr_cols) {
	int i, j, row=0;

	// fill X, Y_left
	for(i=0; i<split_idx; i++) {
		row = idx_array[i];
		for(j=0; j<nr_cols; j++) {
			X_left[i*nr_cols+j]	= X[row*nr_cols+j];
			Y_left[i]			= Y[row];
		}
	}
	// fill X, Y_right
	for(i=split_idx; i<nr_rows; i++) {
		row = idx_array[i];
		for(j=0; j<nr_cols; j++) {
			X_right[(i-split_idx)*nr_cols+j]	= X[row*nr_cols+j];
			Y_right[(i-split_idx)]				= Y[row];
		}
	}
	return;
}












