/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated to match teleop_control/src/teleop_control/lcm_types/body_control_data_lcmt.py
 **/

#ifndef __body_control_data_lcmt_hpp__
#define __body_control_data_lcmt_hpp__

#include <lcm/lcm_coretypes.h>

class body_control_data_lcmt
{
    public:
        /**
         * LCM Type: float[29]
         */
        float      q[29];

        /**
         * LCM Type: float[29]
         */
        float      qd[29];

        /**
         * LCM Type: int64_t
         */
        int64_t    timestamp_us;

    public:
        inline int encode(void *buf, int offset, int maxlen) const;
        inline int getEncodedSize() const;
        inline int decode(const void *buf, int offset, int maxlen);
        inline static int64_t getHash();
        inline static const char* getTypeName();

        inline int _encodeNoHash(void *buf, int offset, int maxlen) const;
        inline int _getEncodedSizeNoHash() const;
        inline int _decodeNoHash(const void *buf, int offset, int maxlen);
        inline static uint64_t _computeHash(const __lcm_hash_ptr *p);
};

int body_control_data_lcmt::encode(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;
    int64_t hash = getHash();

    tlen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if (tlen < 0) return tlen; else pos += tlen;

    tlen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
    if (tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int body_control_data_lcmt::decode(const void *buf, int offset, int maxlen)
{
    int pos = 0, thislen;

    int64_t msg_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (msg_hash != getHash()) return -1;

    thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int body_control_data_lcmt::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t body_control_data_lcmt::getHash()
{
    static int64_t hash = static_cast<int64_t>(_computeHash(NULL));
    return hash;
}

const char* body_control_data_lcmt::getTypeName()
{
    return "body_control_data_lcmt";
}

int body_control_data_lcmt::_encodeNoHash(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->q[0], 29);
    if (tlen < 0) return tlen; else pos += tlen;

    tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->qd[0], 29);
    if (tlen < 0) return tlen; else pos += tlen;

    tlen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &this->timestamp_us, 1);
    if (tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int body_control_data_lcmt::_decodeNoHash(const void *buf, int offset, int maxlen)
{
    int pos = 0, tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->q[0], 29);
    if (tlen < 0) return tlen; else pos += tlen;

    tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->qd[0], 29);
    if (tlen < 0) return tlen; else pos += tlen;

    tlen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &this->timestamp_us, 1);
    if (tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int body_control_data_lcmt::_getEncodedSizeNoHash() const
{
    int enc_size = 0;
    enc_size += __float_encoded_array_size(NULL, 29);
    enc_size += __float_encoded_array_size(NULL, 29);
    enc_size += __int64_t_encoded_array_size(NULL, 1);
    return enc_size;
}

uint64_t body_control_data_lcmt::_computeHash(const __lcm_hash_ptr *)
{
    uint64_t hash = 0xf4f03934711bf8b5ULL;
    return hash;
}

#endif
