#include <stdio.h>
#include "Fibonacci.h"

int main(){
    int num = 0;

    printf("Input a number to calculate:\n");
    scanf("%d", &num);

    printf("result: %d", calc_CUDA_Fibonacci(num));

    getchar();
    getchar();
}