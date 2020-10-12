#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "dataset.h"
#include "output.h"
#include "find_frequent_pairs.h"
#include <inttypes.h>
#include <immintrin.h>
#include <math.h>

int
document_has_word(const dataset *ds, size_t doc_index, size_t voc_index)
{
    // Auxiliary function for `find_pairs_naive_bitmaps`

    uint8_t *column_ptr = get_term_bitmap(ds, voc_index);
    if (doc_index >= ds->num_documents)
    {
        printf("error: doc_index out of bounds %ld/%ld\n", doc_index, ds->num_documents);
        exit(1);
    }

    size_t byte_i = doc_index / 8;
    size_t bit_i = 7 - (doc_index % 8);

    uint8_t b = column_ptr[byte_i];
    return ((b >> bit_i) & 0x1) ? 1 : 0;
}

void
find_pairs_quick_bitmaps(const dataset *ds, output_pairs *op, int threshold)
{
	for (size_t t1 = 0; t1 < ds->vocab_size; ++t1){
		uint8_t *column_ptr1 = get_term_bitmap(ds, t1);
		size_t size= get_term_bitmap_len(ds);
		uint8_t alpha[size];
		int count1=0;
		int n8=(size/32)*32;
		int j=0;
		__m256i g, h;
		for(;j<n8; j+=32){ 
			g = _mm256_loadu_si256 (( __m256i *) &column_ptr1[j]);
			_mm256_storeu_si256 (( __m256i *) &alpha[j] , g );
		}
		for(;j<size;j++){
                        alpha[j]=column_ptr1[j];
                }
		for(int j=0; j<sizeof(alpha);j++){
                        count1 += _mm_popcnt_u32(alpha[j]);
                }
		if(count1>=threshold){
			for (size_t t2 = t1+1; t2 < ds->vocab_size; ++t2){
				uint8_t *column_ptr2 = get_term_bitmap(ds, t2);
				uint8_t beta[size];
				int count2=0;
				int k=0;
				for(;k<n8; k+=32){ 
                        		h = _mm256_loadu_si256 (( __m256i *) &column_ptr2[k]);
                        		_mm256_storeu_si256 (( __m256i *) &beta[k] , h );
                		}
                		for(;k<size;k++){
                        		beta[k]=column_ptr2[k];
                		}
                		for(int k=0; k<sizeof(beta);k++){
                        		count2 += _mm_popcnt_u32(beta[k]);
                		}
				if(count2>=threshold){
	    				uint8_t c[size];
  					int count=0;
					int i=0;
					for(;i<(n8); i+=32){
            				__m256i x = _mm256_loadu_si256 (( __m256i *) &column_ptr1[i]);
            				__m256i y = _mm256_loadu_si256 (( __m256i *) &column_ptr2[i]);
            				__m256i z = _mm256_and_si256 (x , y );
            				_mm256_storeu_si256 (( __m256i *) &c[i] , z );
					}
					for(;i<size;i++){
						c[i]=column_ptr1[i]&column_ptr2[i];
					}
					for(int i=0; i<sizeof(c);i++){
						count += _mm_popcnt_u32(c[i]);
					}
	   				if(count>= threshold){
	   					push_output_pair(op,t1,t2,count);
           				}
				}
			}
		}
	}
}

void
find_pairs_naive_bitmaps(const dataset *ds, output_pairs *op, int threshold)
{
	// This is an example implementation. You don't need to change this, you
	// should implement `find_pairs_quick_*`

	for (size_t t1 = 0; t1 < ds->vocab_size; ++t1)
    	{
        	for (size_t t2 = t1+1; t2 < ds->vocab_size; ++t2)
        	{
            		int count = 0;
            		for (size_t d = 0; d < ds->num_documents; ++d)
            		{
                		int term1_appears_in_doc = document_has_word(ds, d, t1);
                		int term2_appears_in_doc = document_has_word(ds, d, t2);
                		if (term1_appears_in_doc && term2_appears_in_doc)
                		{
                    		++count;
                		}
            		}
            		if (count >= threshold)
                		push_output_pair(op, t1, t2, count);
        	}
    	}
}

void
find_pairs_naive_indexes(const dataset *ds, output_pairs *op, int threshold)
{
    // This is an example implementation. You don't need to change this, you
    // should implement `find_pairs_quick_*`.

    for (size_t t1 = 0; t1 < ds->vocab_size; ++t1)
    {
        const index_list *il1 = get_term_indexes(ds, t1);
        	for (size_t t2 = t1+1; t2 < ds->vocab_size; ++t2)
	        {
            		const index_list *il2 = get_term_indexes(ds, t2);
            		int count = 0;
            		size_t i1 = 0, i2 = 0;
            		for (; i1 < il1->len && i2 < il2->len;)
           		 {
                		size_t x1 = il1->indexes[i1], x2 = il2->indexes[i2];
                		if (x1 == x2) { ++count; ++i1; ++i2; }
                		else if (x1 < x2) { ++i1; }
                		else { ++i2; }
            		}
               		if (count >= threshold)
                		push_output_pair(op, t1, t2, count);
        	
		}
    }	
}

void
find_pairs_quick_indexes(const dataset *ds, output_pairs *op, int threshold)
{
        for (size_t t1 = 0; t1 < ds->vocab_size; ++t1){
                const index_list *il1 = get_term_indexes(ds, t1);
		if(il1->len >= threshold){
                	for (size_t t2 = t1+1; t2 <ds->vocab_size; ++t2){
                        	const index_list *il2 = get_term_indexes(ds, t2);
				if(il2->len >= threshold){
                        		int count = 0;
                        		if(il1->len <= il2->len){
                                		count = BinarySearch(il1,il2);
                        		}else{
                                		count = BinarySearch(il2,il1);
                        		}
                        		if (count >= threshold){
                                		push_output_pair(op, t1, t2, count);
                        		}
                		}
        		}
		}
	}
}

int
BinarySearch(const index_list*small, const index_list*big)
{
        int count =0;
        int min=0;
        int max=(big->len)-1;
        int keymax= small->indexes[max];
        for(int i=0;i< small->len;i++){
                int key=small->indexes[i];
                max=(big->len)-1;
                while (min <=max){
                        int middleIndex =(min+max)/2;
                        int middleValue= big->indexes[middleIndex];
                        if(key == middleValue){
                                count++;
                                min = middleIndex+1;                        
                                break;
                        }else if(middleValue>key){
                                max = middleIndex-1;                            
                        }else if(middleValue<key){
                                min = middleIndex + 1;
                        }
                }
        }
	return count;
}
