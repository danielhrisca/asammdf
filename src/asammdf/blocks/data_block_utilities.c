#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// #define BLKID(a,b,c,d) (((d)<<24|(c)<<16|(b)<<8)|(a)) 
#define DT_ID 1413751587 //BLKID('#', '#', 'D','T') 
#define DZ_ID 1514414883 //BLKID('#', '#', 'D','Z') 
#define DL_ID 1279533859 //BLKID('#', '#', 'D','L') 
#define HL_ID 1279796003 //BLKID('#', '#', 'H','L') 

#define DT_BLOCK 0
#define DZ_BLOCK_DEFLATE 1
#define DZ_BLOCK_TRANSPOSED 2
#define DZ_BLOCK_LZ 3

#define DZ_DEFLATE 0
#define DZ_TRANSPOSED_DEFLATE 1

#define FLAG_DL_EQUAL_LENGHT 1
#define _32MB 33554432

struct COMMON_HEADER
{
  uint32_t id;   
  uint32_t reserved;  
  uint64_t length;   
  uint64_t links_nr;   
};
#define COMMON_SIZE 24

struct COMMON_SHORT_HEADER
{
  uint32_t id;   
  uint32_t reserved;  
  uint64_t length;     
};
#define COMMON_SHORT_SIZE 16

struct DZ_INFO
{
	uint16_t original_type;
  uint8_t zip_type;   
  uint8_t reserved;  
  uint32_t param;   
  uint64_t original_size;   
  uint64_t compressed_size;   
};
#define DZ_INFO_SIZE 24

typedef struct BlockInfo {
  uint64_t address;
  uint64_t original_size;
  uint64_t compressed_size;
  uint64_t block_limit;
  uint32_t param;
  uint8_t block_type;
  struct BlockInfo * next;
} BlockInfo, *PtrBlockInfo;


BlockInfo * data_blocks_info(FILE *file, uint64_t address, uint64_t *total_size, uint64_t record_size, uint64_t *count) {
	if (!address || !(*total_size)) return NULL;
	BlockInfo *out=NULL, *last=NULL, *new_block=NULL;
	struct COMMON_HEADER header;
	struct DZ_INFO dz_info;
	uint64_t size, current_size, bl_address, list_size, current_address;
	uint8_t block_type, flags;
	uint32_t param;
		
	uint64_t read_chunk_size, *bl_addresses;
  if (record_size > _32MB) read_chunk_size = record_size;
  else if (record_size) read_chunk_size = (_32MB / record_size) * record_size;
  else read_chunk_size = _32MB;
     
  if (read_chunk_size > *total_size) read_chunk_size = *total_size;
      
  _fseeki64(file, address, SEEK_SET);
  fread(&header, sizeof(struct COMMON_HEADER), 1, file);
  
  if (header.id == HL_ID) {
		fread(&address, 8, 1, file);
		if (!address) return NULL;
		_fseeki64(file, address, SEEK_SET);
		fread(&header, sizeof(struct COMMON_HEADER), 1, file);
	}
	
  if (header.id == DT_ID) {
  	size = header.length - COMMON_SIZE;
  	if (size) {
  		if (size > total_size) size = total_size;
  		address += COMMON_SIZE;
  		while (size) {
  			if (size > read_chunk_size) {
  				current_size = read_chunk_size;
  			}
  			else {
  				current_size = size;
  			}
				*total_size = *total_size - current_size;
				size -= current_size;
				
  				
				new_block = (PtrBlockInfo) malloc (sizeof(struct BlockInfo));
				new_block->address = address;
				new_block->block_type = DT_BLOCK;
				new_block->original_size = current_size;
				new_block->compressed_size = current_size;
				new_block->block_limit = 0;
				new_block->param = 0;
				new_block->next = NULL;
				
				if (out) {
					last->next = new_block;
					last = new_block;
				}
				else {
					last = new_block;
					out = new_block;
				}
				
				address += current_size;
				
				*count += 1;
  		}
  	}
  }
  else if (header.id == DZ_ID) {
  	fread(&dz_info, sizeof(struct DZ_INFO), 1, file);
  	address += COMMON_SIZE + DZ_INFO_SIZE;
  	if (dz_info.original_size) {
  		if (dz_info.zip_type == DZ_DEFLATE) {
  			block_type = DZ_BLOCK_DEFLATE;
  			param = 0;
  		}
  		else {
  			block_type = DZ_BLOCK_TRANSPOSED;
  			param = dz_info.param;
  		}
  		
  		new_block = (struct BlockInfo *) malloc (sizeof(struct BlockInfo));
			new_block->address = address;
			new_block->block_type = block_type;
			new_block->original_size = dz_info.original_size;
			new_block->compressed_size = dz_info.compressed_size;
			new_block->block_limit = (*total_size < dz_info.original_size) ? *total_size : 0;
			new_block->param = param;
			new_block->next = NULL;
			
			if (out) {
					last->next = new_block;
					last = new_block;
				}
				else {
					last = new_block;
					out = new_block;
				}
			
			if (*total_size < dz_info.original_size) {
				*total_size = 0;
			}
			else {
				*total_size = *total_size - dz_info.original_size;
			}
			
			*count = *count + 1;
  		
  	}
  }
  else if (header.id == DL_ID) {
  	while (address) {
			fread(&address, 8, 1, file);
			list_size = header.links_nr - 1;
			if (list_size <= 1) break;
				
			bl_addresses = (uint64_t *) malloc(8 * list_size);
			fread(bl_addresses, 8, list_size, file);
				
			for (int i=0; i<list_size; i++) {
				if (i%10000 == 0) printf("blk %ld\n", i);
				current_address = bl_addresses[i];
				if (!current_address) continue;
				_fseeki64(file, current_address, SEEK_SET);
				fread(&header, sizeof(struct COMMON_HEADER), 1, file);
				
				current_address += COMMON_SIZE;
				
				if (header.id == DT_ID) {
			  	size = header.length - COMMON_SIZE;
			  	if (size) {
			  		if (size > total_size) size = total_size;
			  		while (size) {
			  			if (size > read_chunk_size) {
			  				current_size = read_chunk_size;
			  			}
			  			else {
			  				current_size = size;
			  			}
							*total_size = *total_size - current_size;
							size -= current_size;
							
			  				
							new_block = (PtrBlockInfo) malloc (sizeof(struct BlockInfo));
							new_block->address = current_address;
							new_block->block_type = DT_BLOCK;
							new_block->original_size = current_size;
							new_block->compressed_size = current_size;
							new_block->block_limit = 0;
							new_block->param = 0;
							new_block->next = NULL;
							
							if (out) {
								last->next = new_block;
								last = new_block;
							}
							else {
								last = new_block;
								out = new_block;
							}
							
							current_address += current_size;
							
							*count = *count + 1;
			  		}
			  	}
			  }
			  else if (header.id == DZ_ID) {
			  	fread(&dz_info, sizeof(struct DZ_INFO), 1, file);
			  	current_address += DZ_INFO_SIZE;
			  	if (dz_info.original_size) {
			  		if (dz_info.zip_type == DZ_DEFLATE) {
			  			block_type = DZ_BLOCK_DEFLATE;
			  			param = 0;
			  		}
			  		else {
			  			block_type = DZ_BLOCK_TRANSPOSED;
			  			param = dz_info.param;
			  		}
			  		
			  		new_block = (struct BlockInfo *) malloc (sizeof(struct BlockInfo));
						new_block->address = current_address;
						new_block->block_type = block_type;
						new_block->original_size = dz_info.original_size;
						new_block->compressed_size = dz_info.compressed_size;
						new_block->block_limit = (*total_size < dz_info.original_size) ? *total_size : 0;
						new_block->param = param;
						new_block->next = NULL;
						
						if (out) {
								last->next = new_block;
								last = new_block;
							}
							else {
								last = new_block;
								out = new_block;
							}
						
						if (*total_size < dz_info.original_size) {
							*total_size = 0;
						}
						else {
							*total_size = *total_size - dz_info.original_size;
						}
						
						*count = *count + 1;
			  		
			  	}
			  }
			}
		}
	}

	return out;
}