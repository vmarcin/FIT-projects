/*
 * crc.c (original)
 *
 *  Created on: Dec 19, 2018
 *      Author: Vladimír Marcin (xmarci10@stud.fit.vutbr.cz)
 */
#include "fsl_crc.h"


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "crc.h"

/**
 * https://www.geeksforgeeks.org/write-an-efficient-c-program-to-reverse-bits-of-a-number/
 */
uint32_t reverseBits( uint32_t num ) {
    uint32_t  numOfBits = sizeof(num) * 8;
    uint32_t reverse_num = 0, i, temp;

    for ( i = 0; i < numOfBits; i++ ) {
        temp = (num & (1 << i));
        if ( temp )
            reverse_num |= (1 << ((numOfBits - 1) - i));
    }

    return reverse_num;
}

uint16_t computeCRC16HWMOD( char *data, uint16_t polynomial ) {
	size_t datalen = strlen(data);
	crc_config_t config;
	CRC_Type *base = CRC0;
	config.reflectIn = false;
	config.reflectOut = false;
    config.polynomial = polynomial;
    config.seed = 0x0U;
    config.complementChecksum = false;
    config.crcBits = kCrcBits16;
    config.crcResult = kCrcFinalChecksum;

    CRC_Init(base, &config); /// enable CRC modul and set 'config'
    CRC_WriteData(base, (uint8_t *) data, datalen); /// write data which we want to encode
    return CRC_Get16bitResult(base); /// get result
}

uint32_t computeCRC32HWMOD( char *data, uint32_t polynomial ) {
	size_t datalen = strlen(data);
	crc_config_t config;
	CRC_Type *base = CRC0;
	config.reflectIn = true;
	config.reflectOut = true;
    config.polynomial = polynomial;
    config.seed = 0xFFFFFFFFU;
    config.complementChecksum = true;
    config.crcBits = kCrcBits32;
    config.crcResult = kCrcFinalChecksum;

    CRC_Init(base, &config); /// enable CRC modul and set 'config'
    CRC_WriteData(base, (uint8_t *) data, datalen); /// write data which we want to encode
    return CRC_Get32bitResult(base); /// get result
}

uint16_t computeCRC16Simple( char *bytes, uint16_t polynomial ) {
    uint16_t crc = 0;

    for ( ; *bytes != '\0'; bytes++ ) {
        crc ^= (uint16_t)(*bytes << 8);

        for ( int i = 0; i < 8; i++ ) {
            if ( (crc & 0x8000) != 0 )
            	crc = (uint16_t)( (crc << 1) ^ polynomial );
            else
            	crc <<= 1;
        }
    }
    return crc;
}

uint32_t computeCRC32Simple(  char *bytes, uint32_t polynomial ) {
    uint32_t crc = 0xFFFFFFFFU;
    polynomial = reverseBits( polynomial );

    for ( ; *bytes != '\0'; bytes++ ) {
        crc ^= (*bytes);
        for ( int i = 0; i < 8; i++ ) {
            if ( crc & 1 )
            	crc = ( (crc >> 1) ^ polynomial );
            else
            	crc >>= 1;
        }
    }
    return ~crc;
}

void calculateTableCRC16( uint16_t polynomial ) {
    uint16_t curByte = 0;

    for ( int divident = 0; divident < 256; divident++, curByte = (divident << 8) ) {
        for ( uint8_t bit = 0; bit < 8; bit++ ) {
            if ( (curByte & 0x8000) != 0 )
            	curByte = (curByte << 1) ^ polynomial;
            else
            	curByte <<= 1;
        }
        crcTable16[divident] = curByte;
    }
}

void calculateTableCRC32( uint32_t polynomial ) {
    polynomial = reverseBits( polynomial );

    for ( uint32_t divident = 0, curByte = 0; divident < 256; divident++, curByte = divident ) {
        for ( uint8_t bit = 0; bit < 8; bit++ ) {
            if ( curByte & 1 )
            	curByte = ( (curByte >> 1) ^ polynomial );
            else
            	curByte >>= 1;
        }
        crcTable32[divident] = curByte;
    }
}

uint16_t computeCRC16Table( char *bytes ) {
    uint16_t crc = 0;

    for ( ; *bytes != '\0'; bytes++ )
        crc = (uint16_t)( crc << 8 ) ^ ( crcTable16[(uint8_t)( (crc >> 8) ^ *bytes )] );
    return crc;
}

uint32_t computeCRC32Table( char *bytes ) {
    uint32_t crc = 0xFFFFFFFFU;

    for ( ; *bytes != '\0'; bytes++ )
        crc = (uint32_t)( crc >> 8 ) ^ ( crcTable32[(uint8_t)(crc ^ *bytes)] );
    return ~crc;
}


