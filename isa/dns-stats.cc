///////////////////////////////////////////////////////////////////////////////////
// School:      Brno University of Technology, Faculty of Information Technology //
// Course:      Network Applications and Network Administration                  //
// Project:     DNS Sniffer                                                      //
// Module:      Stats computation                                                //
// Authors:     Vladimír Marcin     (xmarci10)                                   //
///////////////////////////////////////////////////////////////////////////////////

// C++ header files
#include <arpa/inet.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <algorithm>

// user's header files
#include "dns-stats.h"
#include "base64.h"
#include "dissect.h"

#define DNS_HLEN 12

/**
 * dns header structure
 */
struct dnshdr {
    uint16_t id;
#if __BYTE_ORDER == __LITTLE_ENDIAN
    uint16_t rd:1;
	uint16_t tc:1;
	uint16_t aa:1;
	uint16_t opcode:4;
	uint16_t qr:1;
	uint16_t rcode:4;
	uint16_t z:3;
	uint16_t ra:1;
#elif __BYTE_ORDER == __BIG_ENDIAN
    uint16_t qr:1;
	uint16_t opcode:4;
	uint16_t aa:1;
	uint16_t tc:1;
	uint16_t rd:1;
	uint16_t ra:1;
	uint16_t z:3;
	uint16_t rcode:4;
#else
#  error "Adjust your <bits/endian.h> defines"
# endif
    uint16_t qcount;	/* question count */
	uint16_t ancount;	/* Answer record count */
	uint16_t nscount;	/* Name Server (Autority Record) Count */ 
	uint16_t adcount;	/* Additional Record Count */
};

std::map<std::string, int>stats;
static std::map<uint16_t, std::string>dns_types = {
    {0, "TYPE"},
    {1, "A "},
    {28, "AAAA "},
    {5, "CNAME "},
    {15, "MX "},
    {2, "NS "},
    {6, "SOA "},
    {16, "TXT "},
    {48, "DNSKEY "},
    {47, "NSEC "},
    {43, "DS "},
    {46, "RRSIG "},
    {12, "PTR "},
};

void print_handler(int) {
    pid_t pid;

    /// if fork fails parent process prints stats
    if((pid = fork()) > 0) {
        return;
    }
    print_stats_map();
    if(pid == 0)
        exit(0);
}

void print_stats_map() {
    for(auto it = stats.begin(); it != stats.end(); it++) {
        std::cout << it->first << " " << it->second << "\n";
    }
}

std::string dns_label_to_string(const unsigned char *answer, const unsigned char *dns_header, int *bytes_of_name, uint32_t dns_payload_len) {
    std::string name = {};
    std::string err = "-error-";
    std::string ret;
    char c;
    uint8_t count_of_labels;
    uint16_t offset;

    if(check_pkt_ptr(dns_header, answer, dns_payload_len))
        return err;

    if(((*((uint8_t*)answer)) & 0xC0) == (uint8_t)0xC0) {
        offset = ntohs((*((uint16_t *)answer)));
        offset &= ((uint16_t)0x3FFF);
        *bytes_of_name = 2;
        ret = dns_label_to_string((dns_header + offset), dns_header, bytes_of_name, dns_payload_len);       
        if(!ret.compare(err))
            return err;
        else
            return name + "" + ret;
    } 
    else {
        do {
            count_of_labels = (uint8_t)*(answer++);
            if(!count_of_labels) {
                *bytes_of_name = 1;
                return "<Root>";
            }
            if(check_pkt_ptr(dns_header, (answer + (count_of_labels - 1)), dns_payload_len))
                return err; 
            for(;count_of_labels != 0; count_of_labels --) {
                c = *(answer++);
                name = name + "" + c;
            }
            name = name + ".";
        } while(*(answer) != 0x00 && ((*((uint8_t*)answer)) & 0xC0) != 0xC0);
        if(*(answer) == 0x00) {
            if(!(*bytes_of_name)) {
                *bytes_of_name = name.length() + 1;
            }
            name.pop_back();
            return name;
        }
        else {
            offset = ntohs((*((uint16_t *)answer)));
            offset &= ((uint16_t)0x3FFF);
            if(!(*bytes_of_name)) {
                *bytes_of_name = name.length() + 2;
            }
            ret = dns_label_to_string((dns_header + offset), dns_header, bytes_of_name, dns_payload_len);       
            if(!ret.compare(err))
                return err;
            else
                return name + "" + ret;
        }
    }
    return {}; 
}

unsigned char * skip_queries(const unsigned char *dns_payload, uint16_t qcount, struct packet_payload *payload) {
    uint32_t name_len;
    unsigned char *dns_answers = (unsigned char *)dns_payload;

    while(qcount--) {
        name_len = strlen((char *)dns_answers) + 1;
        dns_answers = (unsigned char *)(dns_answers + (name_len + 4));
        if(check_pkt_ptr(payload->payload, dns_answers, payload->payload_length))
            return NULL;
    }

    return dns_answers;
}

unsigned char * handle_dns_answer(const unsigned char *answer, const unsigned char *dns_header, uint32_t dns_payload_len) {
    int label_length = 0;
    uint16_t answer_type;
    uint16_t rdlength;
    unsigned char *rdata;
    std::string ret = {};
    std::string label;
    std::string err = "-error-";
    std::string rr_answer = {};
    char dbuf[BUFSIZ] = {0,};
    unsigned char *tmp;
    int txt_len;

    label = dns_label_to_string(answer, dns_header, &label_length, dns_payload_len); 
    
    if (err.compare(label))
        ret += label;
    else
        return NULL;

    if(check_pkt_ptr(dns_header, (answer + (label_length + 10)), dns_payload_len))
        return NULL;
    answer_type = ntohs(*((uint16_t *)(answer + label_length)));

    rdlength = ntohs(*((uint16_t *)(answer + (label_length + 8))));
    rdata = (unsigned char *)(answer + (label_length + 10));

    switch (answer_type) {
        case 1: /* A */
            if(check_pkt_ptr(dns_header, (rdata + 3), dns_payload_len))
                return NULL;
            if (inet_ntop(AF_INET, (const char *)rdata, dbuf, INET_ADDRSTRLEN) != NULL) 
                rr_answer += dbuf;
            else 
                return rdata + rdlength;
            break;
        case 2: /* NS */
        case 5: /* CNAME */
        case 12: /* PTR */
            label_length = 0;
            label = dns_label_to_string(rdata, dns_header, &label_length, dns_payload_len);
            if(err.compare(label))
                rr_answer += label;
            else
                return NULL;
            break;
        case 28: /* AAAA */
            if(check_pkt_ptr(dns_header, (rdata + 15), dns_payload_len))
                return NULL;
            if (inet_ntop(AF_INET6, (const char *)rdata, dbuf, INET6_ADDRSTRLEN) != NULL)
                rr_answer += dbuf;
            else 
                return rdata + rdlength;
            break;
        case 15: /* MX */
            if(check_pkt_ptr(dns_header, (rdata + 1), dns_payload_len))
                return NULL;
            snprintf(dbuf, 8,"%d ", ntohs(*(uint16_t *)rdata));
            rr_answer += dbuf;
            label_length = 0;
            label = dns_label_to_string((rdata + 2), dns_header, &label_length, dns_payload_len);
            if(err.compare(label))
                rr_answer += label;
            else 
                return NULL;
            break;
        case 6: /* SOA */
            tmp = rdata;
            label_length = 0;
            label = dns_label_to_string(tmp, dns_header, &label_length, dns_payload_len);
            if(err.compare(label))
                rr_answer += label;
            else
                return NULL;
            tmp += label_length;
            if(check_pkt_ptr(dns_header, tmp, dns_payload_len))
                return NULL;
            label_length = 0;
            label = dns_label_to_string(tmp, dns_header, &label_length, dns_payload_len);
            if(err.compare(label))
                rr_answer = rr_answer + " " + label + " ";
            else
                return NULL;
            tmp += label_length;
            if(check_pkt_ptr(dns_header, (tmp + 19), dns_payload_len))
                return NULL;
            for(int i = 0; i < 5; i++) {
                sprintf(dbuf, "%d ", ntohl(*(uint32_t *)tmp));
                rr_answer += dbuf;
                tmp += 4;
            }
            rr_answer.pop_back();
            break;
        case 99: /* SPF */
        case 16: /* TXT */
            txt_len = *((uint8_t *)rdata);
            tmp = rdata;
            if(check_pkt_ptr(dns_header, (tmp + rdlength), dns_payload_len))
                return NULL;
            while((tmp - rdata) != rdlength) {
                tmp++;
                for(uint8_t i = 0; i < txt_len; i++) {
                    rr_answer += *(char *)tmp;
                    tmp++; 
                }
                txt_len = *((uint8_t *)tmp);
                rr_answer += " ";
            }
            rr_answer.pop_back();
            break;
        case 48: /* DNSKEY */
            if(check_pkt_ptr(dns_header, (rdata + (rdlength - 1)), dns_payload_len ))
                return NULL;
            sprintf(dbuf, "%d", ntohs(*(uint16_t *)rdata));
            rr_answer += dbuf;
            rr_answer += " 3 ";
            sprintf(dbuf, "%d ", *(uint8_t*)(rdata + 3));
            rr_answer += dbuf;
            rr_answer += base64_encode((rdata + 4), (rdlength - 4));
            break;
        case 46: /* RRSIG */
            if(check_pkt_ptr(dns_header, (rdata + (rdlength - 1)), dns_payload_len))
                return NULL;
            struct tm *info;
            time_t t;
            uint16_t type;
            /* type covered */
            type = ntohs(*(uint16_t *)rdata);
            if(dns_types.find(type) != dns_types.end()) {
                rr_answer += dns_types[type];
            }
            else {
                rr_answer += dns_types[0];
                sprintf(dbuf, "%d ", type);
                rr_answer += dbuf;
            }
            /* algorithm, labels */
            for(int i = 2; i < 4; i++) {
                sprintf(dbuf, "%d ", *(uint8_t*)(rdata + i));
                rr_answer += dbuf;
            }
            /* original ttl */
            sprintf(dbuf, "%d ", ntohl(*(uint32_t*)(rdata + 4)));
            rr_answer += dbuf;
            /* signature expiration/inception */
            for(int i = 2; i < 4; i++) {
                t = ntohl(*((time_t *)(rdata + (4*i))));
                info = gmtime(&t);
                strftime(dbuf, 80, "%Y%m%d%H%M%S ", info);
                rr_answer += dbuf;
            }
            /* key tag */
            sprintf(dbuf, "%d ", ntohs(*(uint16_t*)(rdata + 16)));
            rr_answer += dbuf;
            /* signer's name */
            label_length = 0;
            label = dns_label_to_string((rdata + 18), dns_header, &label_length, dns_payload_len);
            if(err.compare(label))
                rr_answer += label;
            else
                return NULL; 
            /* signature */
            rr_answer = rr_answer + " " + base64_encode((rdata + (18 + label_length)), rdlength - (18 + label_length));
            break;
        case 47: /* NSEC */
            uint16_t bitmap_len;
            uint8_t *bitmap;
            /* next domain name */
            label_length = 0;
            label = dns_label_to_string(rdata, dns_header, &label_length, dns_payload_len);
            if(err.compare(label))
                rr_answer += label;
            else 
                return NULL;
            rr_answer += " ";
            /* bit map */
            bitmap_len = ntohs(*(uint16_t*)(rdata + label_length));
            if(check_pkt_ptr(dns_header, (rdata + (label_length + bitmap_len - 1)), dns_payload_len))
                return NULL;
            bitmap = (rdata + (label_length + 2));
            for(uint16_t i = 0; i < (bitmap_len * 8); i++) {
                if(bitmap[i/8] & ((uint8_t)1 << (7 - (i%8)))){
                    if(dns_types.find(i) != dns_types.end()) {
                        rr_answer += dns_types[i];
                    }
                    else {
                        rr_answer += dns_types[0];
                        sprintf(dbuf, "%d ", i);
                        rr_answer += dbuf;
                    }
                }
            }
            rr_answer.pop_back();
            break;
        case 43: /* DS */
            if(check_pkt_ptr(dns_header, (rdata + (rdlength - 1)), dns_payload_len))
                return NULL;
            /* key id */
            sprintf(dbuf, "%d ", ntohs(*(uint16_t *)rdata));
            rr_answer += dbuf;
            /* algorithm */  
            sprintf(dbuf, "%d ", *(uint8_t*)(rdata + 2));
            rr_answer += dbuf;
            /* digest type */ 
            sprintf(dbuf, "%d ", *(uint8_t*)(rdata + 3));
            rr_answer += dbuf;               
            /* digest */    
            tmp = (rdata + 4);
            for(int i = 0; i < (rdlength - 4); i++) {
                sprintf(dbuf, "%02hhX", *(uint8_t*)(tmp++));
                rr_answer += dbuf;
            }
            break;
        default:
            if(check_pkt_ptr(dns_header, (rdata + (rdlength - 1)), dns_payload_len))
                return NULL;
            rr_answer += dns_types[0];
            sprintf(dbuf, "%d \"", answer_type);
            rr_answer += dbuf;
            tmp = rdata;
            for(int i = 0; i < rdlength; i++) {
                sprintf(dbuf, "%02hhX", *(uint8_t*)(tmp++));
                rr_answer += dbuf;
            }
            rr_answer += "\"";
            answer_type = 0;
    }
    if(!answer_type)
        ret = ret + " " + rr_answer;           
    else
        ret = ret + " " + dns_types[answer_type] + "\"" + rr_answer + "\"";
    
   // std::cout << ret << "\n";
    
    stats[ret]++;

    return rdata + rdlength;
}

void process_dns_payload(struct packet_payload *dns_payload) {
    struct dnshdr *dnsh;
    uint16_t ancount;
    uint16_t qcount;
    const unsigned char *dns_data;

    dnsh = (struct dnshdr *)(dns_payload->payload);

    ancount = ntohs(dnsh->ancount);
    qcount = ntohs(dnsh->qcount);

    if (!ancount || dnsh->rcode)
        return;

    /* skip dns header */
    dns_data = dns_payload->payload + DNS_HLEN;
    if(check_pkt_ptr(dns_payload->payload, dns_data, dns_payload->payload_length))
        return;  

    /* skip all queries */
    dns_data = skip_queries(dns_data, qcount, dns_payload);
    if(dns_data == NULL)
        return;

    /* process all answers */
    while (ancount--) {
        dns_data = handle_dns_answer(dns_data, dns_payload->payload, dns_payload->payload_length);
        if(dns_data == NULL)
            return;
        else if(check_pkt_ptr(dns_payload->payload, dns_data, dns_payload->payload_length))
            return;
    }
}