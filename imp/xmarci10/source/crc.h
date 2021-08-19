/*
 * crc.h (original)
 *
 *  Created on: Dec 19, 2018
 *      Author: Vladimír Marcin (xmarci10@stud.fit.vutbr.cz)
 */

#ifndef CRC_H_
#define CRC_H_

#define TAB_SIZE 256

/* CRC Tables */
extern uint16_t crcTable16[TAB_SIZE];
extern uint32_t crcTable32[TAB_SIZE];

/**
 * Function computes CRC16 checksum by HW module
 */
uint16_t computeCRC16HWMOD( char *data, uint16_t polynomial );

/**
 * Function computes CRC32 checksum by HW module
 */
uint32_t computeCRC32HWMOD( char *data, uint32_t polynomial );

/**
 * Simple polynomial CRC16 calculation
 * (algorithm from http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html)
 */
uint16_t computeCRC16Simple( char *bytes, uint16_t polynomial );

/**
 * Simple polynomial CRC32 calculation
 * (algorithm from http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html)
 */
uint32_t computeCRC32Simple(  char *bytes, uint32_t polynomial );

/**
 * CRC16 calculation using lookup table
 * (algorithm from http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html)
 */
uint16_t computeCRC16Table( char *bytes );

/**
 * CRC32 calculation using lookup table
 * (algorithm from http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html)
 */
uint32_t computeCRC32Table( char *bytes );

/**
 * Functions initialize CRC16 lookup table
 * (algorithm from http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html)
 */
void calculateTableCRC16( uint16_t polynomial );

/**
 * Functions initialize CRC32 lookup table
 * (algorithm from http://www.sunshine2k.de/articles/coding/crc/understanding_crc.html)
 */
void calculateTableCRC32( uint32_t polynomial );

#endif /* CRC_H_ */
