#include "turboquant.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace turboquant {
namespace {

void check_cuda(cudaError_t status, const char* label);

// (README) 量化範圍設定: 根據 bit 數決定對稱 signed quantization 可使用的最大整數值。
int qmax_for_bits(int bits) {
    if (bits == 8) return 127;
    if (bits == 4) return 7;
    if (bits == 2) return 1;
    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

int next_power_of_two(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

__host__ __device__ std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// (README) 隨機符號矩陣 D: 用 hash 產生 deterministic 的 Rademacher 正負號來模擬隨機對角矩陣。
__host__ __device__ float rademacher(unsigned seed, std::uint32_t a, std::uint32_t b) {
    std::uint32_t h = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU));
    return (h & 1U) ? 1.0f : -1.0f;
}

__host__ __device__ float uniform01_from_hash(std::uint32_t h) {
    return (static_cast<float>(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

// (README) QJL 隨機投影係數: 用 hash 產生 deterministic Gaussian 樣本供 QJL residual sketch 使用。
__host__ __device__ float gaussian_from_hash(unsigned seed, std::uint32_t a, std::uint32_t b) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    std::uint32_t h0 = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU) ^ 0x243f6a88U);
    std::uint32_t h1 = mix32(seed ^ (a * 0x85ebca6bU) ^ (b * 0xc2b2ae35U) ^ 0x9e3779b9U);
    float u1 = fmaxf(uniform01_from_hash(h0), 1.0e-7f);
    float u2 = uniform01_from_hash(h1);
    return sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
}

void apply_random_sign(std::vector<float>& x, unsigned seed) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] *= rademacher(seed, static_cast<std::uint32_t>(i), 0);
    }
}

// (README) CPU 正規化 Hadamard 轉換: 對向量做 normalized FWHT，對應論文中的隨機旋轉 H。
void fwht_normalized(std::vector<float>& x) {
    const int n = static_cast<int>(x.size());
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                float a = x[i + j];
                float b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));
    for (float& v : x) v *= inv_sqrt_n;
}

void pack_sign_bit(std::vector<std::uint8_t>& signs, int idx, bool positive) {
    if (positive) signs[idx / 8] |= static_cast<std::uint8_t>(1U << (idx % 8));
}

bool unpack_sign_bit(const std::vector<std::uint8_t>& signs, int idx) {
    return ((signs[idx / 8] >> (idx % 8)) & 1U) != 0;
}

std::uint8_t encode_signed_to_unsigned(int q, int bits) {
    if (bits == 8) {
        return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
    }
    if (bits == 4) {
        return static_cast<std::uint8_t>(q + 8);
    }
    if (bits == 2) {
        return static_cast<std::uint8_t>(q + 1);
    }
    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

int decode_unsigned_to_signed(std::uint8_t code, int bits) {
    if (bits == 8) {
        return static_cast<int>(static_cast<std::int8_t>(code));
    }
    if (bits == 4) {
        return static_cast<int>(code & 0x0f) - 8;
    }
    if (bits == 2) {
        return static_cast<int>(code & 0x03) - 1;
    }
    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

// (README) Low-bit payload 打包: 將 4-bit 或 2-bit 量化碼壓進 byte array 以減少通訊 payload。
std::vector<std::uint8_t> pack_codes(const std::vector<std::uint8_t>& codes, int bits) {
    if (bits == 8) return codes;

    if (bits == 4) {
        std::vector<std::uint8_t> packed((codes.size() + 1) / 2, 0);
        for (std::size_t i = 0; i < codes.size(); ++i) {
            std::uint8_t nibble = static_cast<std::uint8_t>(codes[i] & 0x0f);
            if ((i & 1) == 0) {
                packed[i / 2] = nibble;
            } else {
                packed[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
            }
        }
        return packed;
    }
    if (bits == 2) {
        std::vector<std::uint8_t> packed((codes.size() + 3) / 4, 0);
        for (std::size_t i = 0; i < codes.size(); ++i) {
            std::uint8_t two_bits = static_cast<std::uint8_t>(codes[i] & 0x03);
            packed[i / 4] |= static_cast<std::uint8_t>(two_bits << ((i % 4) * 2));
        }
        return packed;
    }

    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

std::uint8_t unpack_code_at(const std::vector<std::uint8_t>& packed, std::size_t i, int bits) {
    if (bits == 8) return packed[i];

    if (bits == 4) {
        std::uint8_t byte = packed[i / 2];
        if ((i & 1) == 0) return static_cast<std::uint8_t>(byte & 0x0f);
        return static_cast<std::uint8_t>((byte >> 4) & 0x0f);
    }
    if (bits == 2) {
        std::uint8_t byte = packed[i / 4];
        return static_cast<std::uint8_t>((byte >> ((i % 4) * 2)) & 0x03);
    }

    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

struct AbsValue {
    __host__ __device__ float operator()(const float& x) const {
        return fabsf(x);
    }
};

struct SquareValue {
    __host__ __device__ float operator()(const float& x) const {
        return x * x;
    }
};

// (README) Lloyd-Max d=256 codebooks: precomputed scalar quantizers for RHT coordinates of unit vectors.
static constexpr int kLloydTqDim = 256;
__device__ __constant__ float d256_b2_centroids[4] = {
    -0.09423680067f, -0.02828811999f, 0.02828811999f, 0.09423680067f
};
__device__ __constant__ float d256_b2_boundaries[5] = {
    -1.0f, -0.06126246033f, -1.214306433e-17f, 0.06126246033f, 1.0f
};
__device__ __constant__ float d256_b3_centroids[8] = {
    -0.133847944f, -0.08375896775f, -0.04716193454f, -0.01529573574f, 0.01529573574f, 0.04716193454f, 0.08375896775f, 0.133847944f
};
__device__ __constant__ float d256_b3_boundaries[9] = {
    -1.0f, -0.1088034559f, -0.06546045115f, -0.03122883514f, 0.0f, 0.03122883514f, 0.06546045115f, 0.1088034559f,
    1.0f
};
__device__ __constant__ float d256_b4_centroids[16] = {
    -0.1693834024f, -0.1285573497f, -0.1006663658f, -0.07821939632f, -0.05870616747f, -0.04092915969f, -0.02418822811f, -0.008004050633f,
    0.008004050633f, 0.02418822811f, 0.04092915969f, 0.05870616747f, 0.07821939632f, 0.1006663658f, 0.1285573497f, 0.1693834024f
};
__device__ __constant__ float d256_b4_boundaries[17] = {
    -1.0f, -0.148970376f, -0.1146118577f, -0.08944288105f, -0.0684627819f, -0.04981766358f, -0.0325586939f, -0.01609613937f,
    -2.602085214e-18f, 0.01609613937f, 0.0325586939f, 0.04981766358f, 0.0684627819f, 0.08944288105f, 0.1146118577f, 0.148970376f,
    1.0f
};
__device__ __constant__ float d256_b5_centroids[32] = {
    -0.1968189485f, -0.1613888613f, -0.1380510167f, -0.1199880961f, -0.1049509203f, -0.09189960948f, -0.08026061443f, -0.06967922569f,
    -0.05991670077f, -0.05080091577f, -0.04220017202f, -0.03400820983f, -0.02613512769f, -0.01850162311f, -0.01103519122f, -0.00366750469f,
    0.00366750469f, 0.01103519122f, 0.01850162311f, 0.02613512769f, 0.03400820983f, 0.04220017202f, 0.05080091577f, 0.05991670077f,
    0.06967922569f, 0.08026061443f, 0.09189960948f, 0.1049509203f, 0.1199880961f, 0.1380510167f, 0.1613888613f, 0.1968189485f
};
__device__ __constant__ float d256_b5_boundaries[33] = {
    -1.0f, -0.1791039049f, -0.149719939f, -0.1290195564f, -0.1124695082f, -0.09842526487f, -0.08608011195f, -0.07496992006f,
    -0.06479796323f, -0.05535880827f, -0.0465005439f, -0.03810419093f, -0.03007166876f, -0.0223183754f, -0.01476840716f, -0.007351347953f,
    -6.071532166e-18f, 0.007351347953f, 0.01476840716f, 0.0223183754f, 0.03007166876f, 0.03810419093f, 0.0465005439f, 0.05535880827f,
    0.06479796323f, 0.07496992006f, 0.08608011195f, 0.09842526487f, 0.1124695082f, 0.1290195564f, 0.149719939f, 0.1791039049f,
    1.0f
};
__device__ __constant__ float d256_b6_centroids[64] = {
    -0.2129712232f, -0.1801739849f, -0.1589073018f, -0.1427070293f, -0.1294465821f, -0.1181471866f, -0.1082729207f, -0.09949614824f,
    -0.09160018648f, -0.0844327429f, -0.07788130515f, -0.07185915045f, -0.06629694947f, -0.06113752121f, -0.05633245908f, -0.05183991486f,
    -0.04762312078f, -0.04364939152f, -0.03988944047f, -0.03631690128f, -0.03290798108f, -0.02964119573f, -0.02649715347f, -0.0234583647f,
    -0.02050906398f, -0.01763503572f, -0.0148234393f, -0.01206263149f, -0.009341986166f, -0.006651711489f, -0.003982665526f, -0.001326170909f,
    0.001326170909f, 0.003982665526f, 0.006651711489f, 0.009341986166f, 0.01206263149f, 0.0148234393f, 0.01763503572f, 0.02050906398f,
    0.0234583647f, 0.02649715347f, 0.02964119573f, 0.03290798108f, 0.03631690128f, 0.03988944047f, 0.04364939152f, 0.04762312078f,
    0.05183991486f, 0.05633245908f, 0.06113752121f, 0.06629694947f, 0.07185915045f, 0.07788130515f, 0.0844327429f, 0.09160018648f,
    0.09949614824f, 0.1082729207f, 0.1181471866f, 0.1294465821f, 0.1427070293f, 0.1589073018f, 0.1801739849f, 0.2129712232f
};
__device__ __constant__ float d256_b6_boundaries[65] = {
    -1.0f, -0.1965726041f, -0.1695406433f, -0.1508071655f, -0.1360768057f, -0.1237968843f, -0.1132100536f, -0.1038845345f,
    -0.09554816736f, -0.08801646469f, -0.08115702403f, -0.0748702278f, -0.06907804996f, -0.06371723534f, -0.05873499014f, -0.05408618697f,
    -0.04973151782f, -0.04563625615f, -0.04176941599f, -0.03810317088f, -0.03461244118f, -0.0312745884f, -0.0280691746f, -0.02497775909f,
    -0.02198371434f, -0.01907204985f, -0.01622923751f, -0.01344303539f, -0.01070230883f, -0.007996848828f, -0.005317188508f, -0.002654418217f,
    -3.024924061e-17f, 0.002654418217f, 0.005317188508f, 0.007996848828f, 0.01070230883f, 0.01344303539f, 0.01622923751f, 0.01907204985f,
    0.02198371434f, 0.02497775909f, 0.0280691746f, 0.0312745884f, 0.03461244118f, 0.03810317088f, 0.04176941599f, 0.04563625615f,
    0.04973151782f, 0.05408618697f, 0.05873499014f, 0.06371723534f, 0.06907804996f, 0.0748702278f, 0.08115702403f, 0.08801646469f,
    0.09554816736f, 0.1038845345f, 0.1132100536f, 0.1237968843f, 0.1360768057f, 0.1508071655f, 0.1695406433f, 0.1965726041f,
    1.0f
};
__device__ __constant__ float d256_b7_centroids[128] = {
    -0.2262967226f, -0.1954226289f, -0.1755995896f, -0.1606363046f, -0.1484970997f, -0.1382450736f, -0.1293669436f, -0.121548549f,
    -0.1145817727f, -0.108320062f, -0.1026549505f, -0.09750273321f, -0.09279648385f, -0.08848107466f, -0.0845099753f, -0.08084314808f,
    -0.07744563963f, -0.07428662322f, -0.07133873558f, -0.06857760598f, -0.0659815095f, -0.06353109907f, -0.06120918608f, -0.05900054986f,
    -0.05689176422f, -0.05487103394f, -0.05292803804f, -0.05105377886f, -0.04924043719f, -0.0474812346f, -0.04577030425f, -0.04410257133f,
    -0.04247364392f, -0.04087971498f, -0.03931747531f, -0.03778403744f, -0.03627686981f, -0.03479374055f, -0.03333266986f, -0.03189189014f,
    -0.03046981274f, -0.02906500051f, -0.02767614514f, -0.02630204849f, -0.02494160722f, -0.02359380007f, -0.02225767717f, -0.0209323511f,
    -0.01961698921f, -0.01831080691f, -0.01701306181f, -0.01572304847f, -0.01444009365f, -0.01316355197f, -0.01189280201f, -0.01062724271f,
    -0.009366290123f, -0.008109374497f, -0.006855937671f, -0.005605430777f, -0.004357312239f, -0.003111046039f, -0.001866100214f, -0.000621945555f,
    0.000621945555f, 0.001866100214f, 0.003111046039f, 0.004357312239f, 0.005605430777f, 0.006855937671f, 0.008109374497f, 0.009366290123f,
    0.01062724271f, 0.01189280201f, 0.01316355197f, 0.01444009365f, 0.01572304847f, 0.01701306181f, 0.01831080691f, 0.01961698921f,
    0.0209323511f, 0.02225767717f, 0.02359380007f, 0.02494160722f, 0.02630204849f, 0.02767614514f, 0.02906500051f, 0.03046981274f,
    0.03189189014f, 0.03333266986f, 0.03479374055f, 0.03627686981f, 0.03778403744f, 0.03931747531f, 0.04087971498f, 0.04247364392f,
    0.04410257133f, 0.04577030425f, 0.0474812346f, 0.04924043719f, 0.05105377886f, 0.05292803804f, 0.05487103394f, 0.05689176422f,
    0.05900054986f, 0.06120918608f, 0.06353109907f, 0.0659815095f, 0.06857760598f, 0.07133873558f, 0.07428662322f, 0.07744563963f,
    0.08084314808f, 0.0845099753f, 0.08848107466f, 0.09279648385f, 0.09750273321f, 0.1026549505f, 0.108320062f, 0.1145817727f,
    0.121548549f, 0.1293669436f, 0.1382450736f, 0.1484970997f, 0.1606363046f, 0.1755995896f, 0.1954226289f, 0.2262967226f
};
__device__ __constant__ float d256_b7_boundaries[129] = {
    -1.0f, -0.2108596758f, -0.1855111093f, -0.1681179471f, -0.1545667021f, -0.1433710866f, -0.1338060086f, -0.1254577463f,
    -0.1180651608f, -0.1114509174f, -0.1054875063f, -0.1000788419f, -0.09514960853f, -0.09063877926f, -0.08649552498f, -0.08267656169f,
    -0.07914439385f, -0.07586613142f, -0.0728126794f, -0.06995817078f, -0.06727955774f, -0.06475630429f, -0.06237014258f, -0.06010486797f,
    -0.05794615704f, -0.05588139908f, -0.05389953599f, -0.05199090845f, -0.05014710803f, -0.0483608359f, -0.04662576943f, -0.04493643779f,
    -0.04328810762f, -0.04167667945f, -0.04009859515f, -0.03855075638f, -0.03703045362f, -0.03553530518f, -0.03406320521f, -0.03261228f,
    -0.03118085144f, -0.02976740663f, -0.02837057283f, -0.02698909681f, -0.02562182786f, -0.02426770365f, -0.02292573862f, -0.02159501413f,
    -0.02027467015f, -0.01896389806f, -0.01766193436f, -0.01636805514f, -0.01508157106f, -0.01380182281f, -0.01252817699f, -0.01126002236f,
    -0.009996766417f, -0.00873783231f, -0.007482656084f, -0.006230684224f, -0.004981371508f, -0.003734179139f, -0.002488573126f, -0.001244022884f,
    -3.106239224e-17f, 0.001244022884f, 0.002488573126f, 0.003734179139f, 0.004981371508f, 0.006230684224f, 0.007482656084f, 0.00873783231f,
    0.009996766417f, 0.01126002236f, 0.01252817699f, 0.01380182281f, 0.01508157106f, 0.01636805514f, 0.01766193436f, 0.01896389806f,
    0.02027467015f, 0.02159501413f, 0.02292573862f, 0.02426770365f, 0.02562182786f, 0.02698909681f, 0.02837057283f, 0.02976740663f,
    0.03118085144f, 0.03261228f, 0.03406320521f, 0.03553530518f, 0.03703045362f, 0.03855075638f, 0.04009859515f, 0.04167667945f,
    0.04328810762f, 0.04493643779f, 0.04662576943f, 0.0483608359f, 0.05014710803f, 0.05199090845f, 0.05389953599f, 0.05588139908f,
    0.05794615704f, 0.06010486797f, 0.06237014258f, 0.06475630429f, 0.06727955774f, 0.06995817078f, 0.0728126794f, 0.07586613142f,
    0.07914439385f, 0.08267656169f, 0.08649552498f, 0.09063877926f, 0.09514960853f, 0.1000788419f, 0.1054875063f, 0.1114509174f,
    0.1180651608f, 0.1254577463f, 0.1338060086f, 0.1433710866f, 0.1545667021f, 0.1681179471f, 0.1855111093f, 0.2108596758f,
    1.0f
};
__device__ __constant__ float d256_b8_centroids[256] = {
    -0.2382761001f, -0.2089773083f, -0.1903064275f, -0.1763069697f, -0.1650213732f, -0.1555486642f, -0.1473949888f, -0.1402577884f,
    -0.1339363787f, -0.1282892493f, -0.1232115449f, -0.1186222938f, -0.1144567602f, -0.1106616725f, -0.1071921515f, -0.1040096854f,
    -0.1010807666f, -0.09837595734f, -0.09586923608f, -0.09353752603f, -0.09136034373f, -0.08931952418f, -0.08739899436f, -0.08558457713f,
    -0.08386381384f, -0.08222579943f, -0.08066102635f, -0.07916123631f, -0.07771927971f, -0.07632898342f, -0.07498502777f, -0.07368283338f,
    -0.0724184585f, -0.07118850698f, -0.06999004681f, -0.06882053885f, -0.06767777513f, -0.06655982607f, -0.06546499561f, -0.06439178352f,
    -0.06333885389f, -0.06230500904f, -0.06128916801f, -0.06029034891f, -0.05930765452f, -0.05834026058f, -0.05738740623f, -0.05644838623f,
    -0.05552254462f, -0.05460926946f, -0.05370798845f, -0.0528181653f, -0.05193929651f, -0.05107090871f, -0.05021255624f, -0.04936381904f,
    -0.04852430075f, -0.04769362712f, -0.04687144444f, -0.0460574183f, -0.04525123238f, -0.04445258751f, -0.04366120071f, -0.04287680438f,
    -0.04209914558f, -0.04132798525f, -0.0405630975f, -0.03980426882f, -0.03905129732f, -0.03830399178f, -0.03756217077f, -0.03682566161f,
    -0.0360942994f, -0.03536792588f, -0.03464638848f, -0.03392953928f, -0.03321723411f, -0.0325093318f, -0.03180569347f, -0.03110618206f,
    -0.03041066204f, -0.02971899922f, -0.02903106074f, -0.02834671527f, -0.02766583319f, -0.02698828699f, -0.02631395162f, -0.02564270492f,
    -0.02497442793f, -0.02430900534f, -0.02364632565f, -0.02298628144f, -0.02232876946f, -0.02167369066f, -0.02102095012f, -0.02037045692f,
    -0.01972212396f, -0.01907586767f, -0.01843160776f, -0.01778926684f, -0.01714877012f, -0.01651004503f, -0.01587302086f, -0.01523762845f,
    -0.01460379984f, -0.01397146802f, -0.01334056665f, -0.01271102988f, -0.0120827922f, -0.01145578836f, -0.01082995329f, -0.01020522215f,
    -0.009581530291f, -0.008958813359f, -0.008337007319f, -0.007716048503f, -0.007095873633f, -0.006476419809f, -0.005857624468f, -0.005239425306f,
    -0.004621760182f, -0.004004566996f, -0.003387783577f, -0.002771347579f, -0.002155196406f, -0.001539267175f, -0.0009234967204f, -0.0003078216371f,
    0.0003078216371f, 0.0009234967204f, 0.001539267175f, 0.002155196406f, 0.002771347579f, 0.003387783577f, 0.004004566996f, 0.004621760182f,
    0.005239425306f, 0.005857624468f, 0.006476419809f, 0.007095873633f, 0.007716048503f, 0.008337007319f, 0.008958813359f, 0.009581530291f,
    0.01020522215f, 0.01082995329f, 0.01145578836f, 0.0120827922f, 0.01271102988f, 0.01334056665f, 0.01397146802f, 0.01460379984f,
    0.01523762845f, 0.01587302086f, 0.01651004503f, 0.01714877012f, 0.01778926684f, 0.01843160776f, 0.01907586767f, 0.01972212396f,
    0.02037045692f, 0.02102095012f, 0.02167369066f, 0.02232876946f, 0.02298628144f, 0.02364632565f, 0.02430900534f, 0.02497442793f,
    0.02564270492f, 0.02631395162f, 0.02698828699f, 0.02766583319f, 0.02834671527f, 0.02903106074f, 0.02971899922f, 0.03041066204f,
    0.03110618206f, 0.03180569347f, 0.0325093318f, 0.03321723411f, 0.03392953928f, 0.03464638848f, 0.03536792588f, 0.0360942994f,
    0.03682566161f, 0.03756217077f, 0.03830399178f, 0.03905129732f, 0.03980426882f, 0.0405630975f, 0.04132798525f, 0.04209914558f,
    0.04287680438f, 0.04366120071f, 0.04445258751f, 0.04525123238f, 0.0460574183f, 0.04687144444f, 0.04769362712f, 0.04852430075f,
    0.04936381904f, 0.05021255624f, 0.05107090871f, 0.05193929651f, 0.0528181653f, 0.05370798845f, 0.05460926946f, 0.05552254462f,
    0.05644838623f, 0.05738740623f, 0.05834026058f, 0.05930765452f, 0.06029034891f, 0.06128916801f, 0.06230500904f, 0.06333885389f,
    0.06439178352f, 0.06546499561f, 0.06655982607f, 0.06767777513f, 0.06882053885f, 0.06999004681f, 0.07118850698f, 0.0724184585f,
    0.07368283338f, 0.07498502777f, 0.07632898342f, 0.07771927971f, 0.07916123631f, 0.08066102635f, 0.08222579943f, 0.08386381384f,
    0.08558457713f, 0.08739899436f, 0.08931952418f, 0.09136034373f, 0.09353752603f, 0.09586923608f, 0.09837595734f, 0.1010807666f,
    0.1040096854f, 0.1071921515f, 0.1106616725f, 0.1144567602f, 0.1186222938f, 0.1232115449f, 0.1282892493f, 0.1339363787f,
    0.1402577884f, 0.1473949888f, 0.1555486642f, 0.1650213732f, 0.1763069697f, 0.1903064275f, 0.2089773083f, 0.2382761001f
};
__device__ __constant__ float d256_b8_boundaries[257] = {
    -1.0f, -0.2236267042f, -0.1996418679f, -0.1833066986f, -0.1706641715f, -0.1602850187f, -0.1514718265f, -0.1438263886f,
    -0.1370970835f, -0.131112814f, -0.1257503971f, -0.1209169193f, -0.116539527f, -0.1125592163f, -0.108926912f, -0.1056009185f,
    -0.102545226f, -0.09972836195f, -0.09712259671f, -0.09470338106f, -0.09244893488f, -0.09033993395f, -0.08835925927f, -0.08649178574f,
    -0.08472419549f, -0.08304480664f, -0.08144341289f, -0.07991113133f, -0.07844025801f, -0.07702413157f, -0.0756570056f, -0.07433393058f,
    -0.07305064594f, -0.07180348274f, -0.0705892769f, -0.06940529283f, -0.06824915699f, -0.0671188006f, -0.06601241084f, -0.06492838957f,
    -0.06386531871f, -0.06282193147f, -0.06179708853f, -0.06078975846f, -0.05979900172f, -0.05882395755f, -0.05786383341f, -0.05691789623f,
    -0.05598546542f, -0.05506590704f, -0.05415862896f, -0.05326307688f, -0.05237873091f, -0.05150510261f, -0.05064173248f, -0.04978818764f,
    -0.04894405989f, -0.04810896394f, -0.04728253578f, -0.04646443137f, -0.04565432534f, -0.04485190995f, -0.04405689411f, -0.04326900255f,
    -0.04248797498f, -0.04171356542f, -0.04094554137f, -0.04018368316f, -0.03942778307f, -0.03867764455f, -0.03793308127f, -0.03719391619f,
    -0.03645998051f, -0.03573111264f, -0.03500715718f, -0.03428796388f, -0.0335733867f, -0.03286328296f, -0.03215751263f, -0.03145593776f,
    -0.03075842205f, -0.03006483063f, -0.02937502998f, -0.02868888801f, -0.02800627423f, -0.02732706009f, -0.0266511193f, -0.02597832827f,
    -0.02530856642f, -0.02464171664f, -0.02397766549f, -0.02331630354f, -0.02265752545f, -0.02200123006f, -0.02134732039f, -0.02069570352f,
    -0.02004629044f, -0.01939899582f, -0.01875373772f, -0.0181104373f, -0.01746901848f, -0.01682940758f, -0.01619153294f, -0.01555532465f,
    -0.01492071415f, -0.01428763393f, -0.01365601734f, -0.01302579826f, -0.01239691104f, -0.01176929028f, -0.01114287083f, -0.01051758772f,
    -0.009893376221f, -0.009270171825f, -0.008647910339f, -0.008026527911f, -0.007405961068f, -0.006786146721f, -0.006167022138f, -0.005548524887f,
    -0.004930592744f, -0.004313163589f, -0.003696175287f, -0.003079565578f, -0.002463271992f, -0.001847231791f, -0.001231381948f, -0.0006156591788f,
    1.840433188e-17f, 0.0006156591788f, 0.001231381948f, 0.001847231791f, 0.002463271992f, 0.003079565578f, 0.003696175287f, 0.004313163589f,
    0.004930592744f, 0.005548524887f, 0.006167022138f, 0.006786146721f, 0.007405961068f, 0.008026527911f, 0.008647910339f, 0.009270171825f,
    0.009893376221f, 0.01051758772f, 0.01114287083f, 0.01176929028f, 0.01239691104f, 0.01302579826f, 0.01365601734f, 0.01428763393f,
    0.01492071415f, 0.01555532465f, 0.01619153294f, 0.01682940758f, 0.01746901848f, 0.0181104373f, 0.01875373772f, 0.01939899582f,
    0.02004629044f, 0.02069570352f, 0.02134732039f, 0.02200123006f, 0.02265752545f, 0.02331630354f, 0.02397766549f, 0.02464171664f,
    0.02530856642f, 0.02597832827f, 0.0266511193f, 0.02732706009f, 0.02800627423f, 0.02868888801f, 0.02937502998f, 0.03006483063f,
    0.03075842205f, 0.03145593776f, 0.03215751263f, 0.03286328296f, 0.0335733867f, 0.03428796388f, 0.03500715718f, 0.03573111264f,
    0.03645998051f, 0.03719391619f, 0.03793308127f, 0.03867764455f, 0.03942778307f, 0.04018368316f, 0.04094554137f, 0.04171356542f,
    0.04248797498f, 0.04326900255f, 0.04405689411f, 0.04485190995f, 0.04565432534f, 0.04646443137f, 0.04728253578f, 0.04810896394f,
    0.04894405989f, 0.04978818764f, 0.05064173248f, 0.05150510261f, 0.05237873091f, 0.05326307688f, 0.05415862896f, 0.05506590704f,
    0.05598546542f, 0.05691789623f, 0.05786383341f, 0.05882395755f, 0.05979900172f, 0.06078975846f, 0.06179708853f, 0.06282193147f,
    0.06386531871f, 0.06492838957f, 0.06601241084f, 0.0671188006f, 0.06824915699f, 0.06940529283f, 0.0705892769f, 0.07180348274f,
    0.07305064594f, 0.07433393058f, 0.0756570056f, 0.07702413157f, 0.07844025801f, 0.07991113133f, 0.08144341289f, 0.08304480664f,
    0.08472419549f, 0.08649178574f, 0.08835925927f, 0.09033993395f, 0.09244893488f, 0.09470338106f, 0.09712259671f, 0.09972836195f,
    0.102545226f, 0.1056009185f, 0.108926912f, 0.1125592163f, 0.116539527f, 0.1209169193f, 0.1257503971f, 0.131112814f,
    0.1370970835f, 0.1438263886f, 0.1514718265f, 0.1602850187f, 0.1706641715f, 0.1833066986f, 0.1996418679f, 0.2236267042f,
    1.0f
};

__host__ __device__ bool is_supported_lloyd_tq_bits(int bits) {
    return bits >= 2 && bits <= 8;
}

std::size_t lloyd_tq_code_bytes(int rows, int cols, int bits) {
    if (rows != kLloydTqDim) {
        throw std::runtime_error("mode=tq Lloyd-Max codebook path requires vector dimension d=256.");
    }
    if (!is_supported_lloyd_tq_bits(bits)) {
        throw std::runtime_error("mode=tq Lloyd-Max codebook path requires bits in [2, 8].");
    }
    const std::size_t bit_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
        static_cast<std::size_t>(bits);
    return ((bit_count + 31U) / 32U) * sizeof(unsigned int);
}

__device__ const float* lloyd_tq_centroids_for_bits(int bits) {
    switch (bits) {
        case 2: return d256_b2_centroids;
        case 3: return d256_b3_centroids;
        case 4: return d256_b4_centroids;
        case 5: return d256_b5_centroids;
        case 6: return d256_b6_centroids;
        case 7: return d256_b7_centroids;
        case 8: return d256_b8_centroids;
        default: return nullptr;
    }
}

__device__ const float* lloyd_tq_boundaries_for_bits(int bits) {
    switch (bits) {
        case 2: return d256_b2_boundaries;
        case 3: return d256_b3_boundaries;
        case 4: return d256_b4_boundaries;
        case 5: return d256_b5_boundaries;
        case 6: return d256_b6_boundaries;
        case 7: return d256_b7_boundaries;
        case 8: return d256_b8_boundaries;
        default: return nullptr;
    }
}

__device__ int lloyd_tq_bucket(float value, int bits) {
    const int levels = 1 << bits;
    const float* boundaries = lloyd_tq_boundaries_for_bits(bits);
    if (!boundaries || isnan(value)) return 0;
    if (value >= boundaries[levels - 1]) return levels - 1;
    for (int i = 0; i < levels; ++i) {
        if (value < boundaries[i + 1]) return i;
    }
    return levels - 1;
}

__device__ void bitpack_write_code(std::uint8_t* packed, std::size_t idx, int bits, int code) {
    unsigned int* words = reinterpret_cast<unsigned int*>(packed);
    const std::size_t bit_offset = idx * static_cast<std::size_t>(bits);
    const std::size_t word_idx = bit_offset >> 5;
    const int shift = static_cast<int>(bit_offset & 31U);
    const unsigned int mask = (bits == 32) ? 0xffffffffU : ((1U << bits) - 1U);
    const unsigned int value = static_cast<unsigned int>(code) & mask;
    atomicOr(words + word_idx, value << shift);
    if (shift + bits > 32) {
        atomicOr(words + word_idx + 1, value >> (32 - shift));
    }
}

__device__ int bitpack_read_code(const std::uint8_t* packed, std::size_t idx, int bits) {
    const unsigned int* words = reinterpret_cast<const unsigned int*>(packed);
    const std::size_t bit_offset = idx * static_cast<std::size_t>(bits);
    const std::size_t word_idx = bit_offset >> 5;
    const int shift = static_cast<int>(bit_offset & 31U);
    const unsigned int mask = (bits == 32) ? 0xffffffffU : ((1U << bits) - 1U);
    unsigned int value = words[word_idx] >> shift;
    if (shift + bits > 32) {
        value |= words[word_idx + 1] << (32 - shift);
    }
    return static_cast<int>(value & mask);
}



// (README) 8-bit 均勻量化: 將 FP32 數值依照 shared scale 四捨五入到 int8 範圍。
__global__ void quantize_int8_kernel(
    const float* values,
    std::uint8_t* codes,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int q = static_cast<int>(nearbyintf(values[idx] / scale));
    q = max(-127, min(127, q));
    codes[idx] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

// (README) 4-bit 均勻量化與打包: 將 FP32 數值量化到 signed 4-bit 並把兩個 code 打包成一個 byte。
__global__ void quantize_int4_pack_kernel(
    const float* values,
    std::uint8_t* packed_codes,
    std::size_t count,
    float scale) {
    const std::size_t out_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t i0 = out_idx * 2;
    if (i0 >= count) return;

    int q0 = static_cast<int>(nearbyintf(values[i0] / scale));
    q0 = max(-7, min(7, q0));
    std::uint8_t c0 = static_cast<std::uint8_t>((q0 + 8) & 0x0f);

    std::uint8_t c1 = 0;
    if (i0 + 1 < count) {
        int q1 = static_cast<int>(nearbyintf(values[i0 + 1] / scale));
        q1 = max(-7, min(7, q1));
        c1 = static_cast<std::uint8_t>((q1 + 8) & 0x0f);
    }

    packed_codes[out_idx] = static_cast<std::uint8_t>(c0 | (c1 << 4));
}

// (README) 2-bit 均勻量化與打包: 將 FP32 數值量化到 signed 2-bit 並把四個 code 打包成一個 byte。
__global__ void quantize_int2_pack_kernel(
    const float* values,
    std::uint8_t* packed_codes,
    std::size_t count,
    float scale) {
    const std::size_t out_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t i0 = out_idx * 4;
    if (i0 >= count) return;

    std::uint8_t packed = 0;
    for (int lane = 0; lane < 4; ++lane) {
        const std::size_t i = i0 + lane;
        if (i >= count) break;
        int q = static_cast<int>(nearbyintf(values[i] / scale));
        q = max(-1, min(1, q));
        std::uint8_t code = static_cast<std::uint8_t>((q + 1) & 0x03);
        packed |= static_cast<std::uint8_t>(code << (lane * 2));
    }
    packed_codes[out_idx] = packed;
}

__global__ void copy_pad_random_sign_kernel(
    const float* values,
    float* padded,
    std::size_t count,
    std::size_t padded_count,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= padded_count) return;
    float v = (idx < count) ? values[idx] : 0.0f;
    padded[idx] = v * rademacher(seed, static_cast<std::uint32_t>(idx), 0);
}

// (README) Column-wise 隨機符號與 padding: 對每個 column 套用 D 並補零到 Hadamard 需要的 2 的冪次長度。
__global__ void copy_pad_random_sign_columns_kernel(
    const float* values,
    float* padded,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(padded_rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(padded_rows));
    float v = (row < rows) ? values[static_cast<std::size_t>(col) * rows + row] : 0.0f;
    padded[idx] = v * rademacher(seed, static_cast<std::uint32_t>(row),
                                 static_cast<std::uint32_t>(col));
}

__global__ void initialize_column_signs_kernel(
    signed char* signs,
    int padded_rows,
    int cols,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(padded_rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(padded_rows));
    signs[idx] = (rademacher(seed, static_cast<std::uint32_t>(row),
                             static_cast<std::uint32_t>(col)) > 0.0f) ? 1 : -1;
}

__global__ void copy_pad_apply_sign_columns_kernel(
    const float* values,
    const signed char* signs,
    float* padded,
    int rows,
    int cols,
    int padded_rows) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(padded_rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(padded_rows));
    float v = (row < rows) ? values[static_cast<std::size_t>(col) * rows + row] : 0.0f;
    padded[idx] = v * static_cast<float>(signs[idx]);
}

__global__ void apply_random_sign_truncate_kernel(
    const float* padded,
    float* values,
    std::size_t count,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    values[idx] = padded[idx] * rademacher(seed, static_cast<std::uint32_t>(idx), 0);
}

__global__ void apply_random_sign_truncate_add_kernel(
    const float* padded,
    float* values,
    std::size_t count,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    values[idx] += padded[idx] * rademacher(seed, static_cast<std::uint32_t>(idx), 0);
}

__global__ void apply_random_sign_truncate_columns_add_kernel(
    const float* padded,
    float* values,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    values[idx] += padded[static_cast<std::size_t>(col) * padded_rows + row] *
                   rademacher(seed, static_cast<std::uint32_t>(row),
                              static_cast<std::uint32_t>(col));
}

__global__ void apply_sign_mask_truncate_columns_add_kernel(
    const float* padded,
    const signed char* signs,
    float* values,
    int rows,
    int cols,
    int padded_rows) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    const std::size_t padded_idx = static_cast<std::size_t>(col) * padded_rows + row;
    values[idx] += padded[padded_idx] * static_cast<float>(signs[padded_idx]);
}

__global__ void add_plain_kernel(float* dst, const float* src, std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] += src[idx];
}

__global__ void residual_kernel(
    const float* values,
    const float* reconstructed,
    float* residual,
    std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    residual[idx] = values[idx] - reconstructed[idx];
}

// (README) QJL residual sketch: 計算 residual 和 hash Gaussian sketch 向量的分段內積。
__global__ void qjl_dot_partial_kernel(
    const float* residual,
    float* partials,
    std::size_t count,
    int qjl_dim,
    int blocks_per_sketch,
    unsigned seed) {
    extern __shared__ float smem[];
    const int s = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (std::size_t j = static_cast<std::size_t>(b) * blockDim.x + tid;
         j < count;
         j += static_cast<std::size_t>(blocks_per_sketch) * blockDim.x) {
        sum += gaussian_from_hash(seed, static_cast<std::uint32_t>(s),
                                  static_cast<std::uint32_t>(j)) * residual[j];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        partials[static_cast<std::size_t>(s) * blocks_per_sketch + b] = smem[0];
    }
}

// (README) QJL sketch 符號壓縮: 將每個 sketch 內積的正負號存成 residual correction 的 compact 表示。
__global__ void qjl_pack_signs_kernel(
    const float* partials,
    int* signs,
    int qjl_dim,
    int blocks_per_sketch) {
    extern __shared__ float smem[];
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < blocks_per_sketch; i += blockDim.x) {
        sum += partials[static_cast<std::size_t>(s) * blocks_per_sketch + i];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        signs[s] = (smem[0] >= 0.0f) ? 1 : 0;
    }
}

// (README) QJL residual 重建: 用 sketch 符號和 hash Gaussian 基底近似補回 quantization residual。
__global__ void qjl_reconstruct_kernel(
    float* reconstructed,
    const int* signs,
    std::size_t count,
    int qjl_dim,
    float coeff,
    unsigned seed) {
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= count) return;
    float accum = 0.0f;
    for (int s = 0; s < qjl_dim; ++s) {
        const float sign = signs[s] ? 1.0f : -1.0f;
        accum += sign * gaussian_from_hash(seed, static_cast<std::uint32_t>(s),
                                           static_cast<std::uint32_t>(j));
    }
    reconstructed[j] += coeff * accum;
}

// (README) FWHT 單階段 butterfly: 執行 Hadamard transform 的一層加減 butterfly。
__global__ void fwht_stage_kernel(float* values, std::size_t count, int len) {
    const std::size_t pair_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pairs = count / 2;
    if (pair_idx >= pairs) return;
    const std::size_t block = pair_idx / static_cast<std::size_t>(len);
    const std::size_t offset = pair_idx % static_cast<std::size_t>(len);
    const std::size_t i0 = block * static_cast<std::size_t>(len) * 2 + offset;
    const std::size_t i1 = i0 + static_cast<std::size_t>(len);
    float a = values[i0];
    float b = values[i1];
    values[i0] = a + b;
    values[i1] = a - b;
}

// (README) Column-wise FWHT 單階段 butterfly: 對多個 column 各自執行 Hadamard transform 的一層 butterfly。
__global__ void fwht_columns_stage_kernel(float* values, int padded_rows, int cols, int len) {
    const std::size_t pair_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pairs_per_col = static_cast<std::size_t>(padded_rows) / 2;
    const std::size_t total_pairs = pairs_per_col * cols;
    if (pair_idx >= total_pairs) return;
    const int col = static_cast<int>(pair_idx / pairs_per_col);
    const std::size_t pair_in_col = pair_idx % pairs_per_col;
    const std::size_t block = pair_in_col / static_cast<std::size_t>(len);
    const std::size_t offset = pair_in_col % static_cast<std::size_t>(len);
    const std::size_t base = static_cast<std::size_t>(col) * padded_rows;
    const std::size_t i0 = base + block * static_cast<std::size_t>(len) * 2 + offset;
    const std::size_t i1 = i0 + static_cast<std::size_t>(len);
    float a = values[i0];
    float b = values[i1];
    values[i0] = a + b;
    values[i1] = a - b;
}

// (README) Column-wise TQ 前向旋轉: 對每個 column 執行 D 隨機符號與 normalized Hadamard 旋轉。
__global__ void column_tq_forward_fwht_kernel(
    const float* values,
    float* transformed,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= padded_rows) return;

    float v = 0.0f;
    if (row < rows) {
        v = values[static_cast<std::size_t>(col) * rows + row] *
            rademacher(seed, static_cast<std::uint32_t>(row),
                       static_cast<std::uint32_t>(col));
    }
    smem[row] = v;
    __syncthreads();

    for (int len = 1; len < padded_rows; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < padded_rows) {
            const float a = smem[i0];
            const float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    const float inv_sqrt_n = rsqrtf(static_cast<float>(padded_rows));
    transformed[static_cast<std::size_t>(col) * padded_rows + row] = smem[row] * inv_sqrt_n;
}

// (README) Column-wise TQ 解碼累加: 將 low-bit code 解量化後做 inverse Hadamard 與 D，並直接加到 accumulator。
__global__ void column_tq_dequant_inverse_fwht_add_kernel(
    const std::uint8_t* codes,
    float* values,
    int rows,
    int cols,
    int padded_rows,
    int bits,
    float scale,
    unsigned seed) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= padded_rows) return;

    const std::size_t idx = static_cast<std::size_t>(col) * padded_rows + row;
    int q = 0;
    if (bits == 8) {
        q = static_cast<int>(static_cast<std::int8_t>(codes[idx]));
    } else if (bits == 4) {
        const std::uint8_t byte = codes[idx / 2];
        const std::uint8_t code = ((idx & 1) == 0) ?
            static_cast<std::uint8_t>(byte & 0x0f) :
            static_cast<std::uint8_t>((byte >> 4) & 0x0f);
        q = static_cast<int>(code) - 8;
    } else {
        const std::uint8_t byte = codes[idx / 4];
        const std::uint8_t code =
            static_cast<std::uint8_t>((byte >> ((idx & 3) * 2)) & 0x03);
        q = static_cast<int>(code) - 1;
    }
    smem[row] = static_cast<float>(q) * scale;
    __syncthreads();

    for (int len = 1; len < padded_rows; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < padded_rows) {
            const float a = smem[i0];
            const float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    if (row < rows) {
        const float inv_sqrt_n = rsqrtf(static_cast<float>(padded_rows));
        values[static_cast<std::size_t>(col) * rows + row] +=
            smem[row] * inv_sqrt_n *
            rademacher(seed, static_cast<std::uint32_t>(row),
                       static_cast<std::uint32_t>(col));
    }
}

// (README) Column-wise TQ 解碼儲存: 將 transformed buffer 做 inverse Hadamard 與 D 後寫回 reconstructed matrix。
__global__ void column_tq_inverse_fwht_store_kernel(
    const float* transformed,
    float* values,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= padded_rows) return;

    smem[row] = transformed[static_cast<std::size_t>(col) * padded_rows + row];
    __syncthreads();

    for (int len = 1; len < padded_rows; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < padded_rows) {
            const float a = smem[i0];
            const float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    if (row < rows) {
        const float inv_sqrt_n = rsqrtf(static_cast<float>(padded_rows));
        values[static_cast<std::size_t>(col) * rows + row] =
            smem[row] * inv_sqrt_n *
            rademacher(seed, static_cast<std::uint32_t>(row),
                       static_cast<std::uint32_t>(col));
    }
}

// (README) Column-wise Lloyd-Max TQ 前向壓縮: 對每個 256 維向量做 norm normalization、RHT 與 Lloyd-Max bucket index bit-pack。
__global__ void column_tq_lloyd_forward_kernel(
    const float* values,
    std::uint8_t* codes,
    float* norms,
    int rows,
    int cols,
    int bits,
    unsigned seed,
    float eps) {
    extern __shared__ float smem[];
    float* x = smem;
    float* sum = smem + kLloydTqDim;
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= kLloydTqDim) return;

    const float v = (row < rows) ? values[static_cast<std::size_t>(col) * rows + row] : 0.0f;
    x[row] = v;
    sum[row] = v * v;
    __syncthreads();

    for (int stride = kLloydTqDim >> 1; stride > 0; stride >>= 1) {
        if (row < stride) sum[row] += sum[row + stride];
        __syncthreads();
    }

    const float norm = sqrtf(fmaxf(sum[0], 0.0f));
    if (row == 0) norms[col] = norm;
    const float inv_norm = 1.0f / (norm + eps);
    x[row] = ((row < rows) ? v * inv_norm : 0.0f) *
             rademacher(seed, static_cast<std::uint32_t>(row),
                        static_cast<std::uint32_t>(col));
    __syncthreads();

    for (int len = 1; len < kLloydTqDim; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < kLloydTqDim) {
            const float a = x[i0];
            const float b = x[i1];
            x[i0] = a + b;
            x[i1] = a - b;
        }
        __syncthreads();
    }

    const float y = x[row] * rsqrtf(static_cast<float>(kLloydTqDim));
    const int code = lloyd_tq_bucket(y, bits);
    bitpack_write_code(codes, static_cast<std::size_t>(col) * kLloydTqDim + row, bits, code);
}

// (README) Column-wise Lloyd-Max TQ 解碼累加: 將 bit-packed index 查表成 centroid，inverse RHT 後乘回原向量 norm 並加到 accumulator。
__global__ void column_tq_lloyd_inverse_add_kernel(
    const std::uint8_t* codes,
    const float* norms,
    float* values,
    int rows,
    int cols,
    int bits,
    unsigned seed) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= kLloydTqDim) return;

    const int code = bitpack_read_code(codes, static_cast<std::size_t>(col) * kLloydTqDim + row, bits);
    const float* centroids = lloyd_tq_centroids_for_bits(bits);
    smem[row] = centroids ? centroids[code] : 0.0f;
    __syncthreads();

    for (int len = 1; len < kLloydTqDim; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < kLloydTqDim) {
            const float a = smem[i0];
            const float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    if (row < rows) {
        const float inv_sqrt_n = rsqrtf(static_cast<float>(kLloydTqDim));
        values[static_cast<std::size_t>(col) * rows + row] +=
            norms[col] * smem[row] * inv_sqrt_n *
            rademacher(seed, static_cast<std::uint32_t>(row),
                       static_cast<std::uint32_t>(col));
    }
}

__global__ void scale_kernel(float* values, std::size_t count, float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    values[idx] *= scale;
}

__global__ void dequantize_int8_kernel(
    const std::uint8_t* codes,
    float* values,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int q = static_cast<int>(static_cast<std::int8_t>(codes[idx]));
    values[idx] = static_cast<float>(q) * scale;
}

__global__ void dequantize_int4_pack_kernel(
    const std::uint8_t* packed_codes,
    float* values,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    std::uint8_t byte = packed_codes[idx / 2];
    std::uint8_t code = ((idx & 1) == 0) ?
        static_cast<std::uint8_t>(byte & 0x0f) :
        static_cast<std::uint8_t>((byte >> 4) & 0x0f);
    int q = static_cast<int>(code) - 8;
    values[idx] = static_cast<float>(q) * scale;
}

__global__ void dequantize_int2_pack_kernel(
    const std::uint8_t* packed_codes,
    float* values,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    std::uint8_t byte = packed_codes[idx / 4];
    std::uint8_t code = static_cast<std::uint8_t>((byte >> ((idx % 4) * 2)) & 0x03);
    int q = static_cast<int>(code) - 1;
    values[idx] = static_cast<float>(q) * scale;
}

// (README) GPU 正規化 Hadamard 轉換: 在 device buffer 上執行 normalized FWHT，對應論文中的 H 旋轉。
void fwht_normalized_device(float* d_values, std::size_t count, cudaStream_t stream) {
    const int threads = 256;
    const std::size_t pairs = count / 2;
    for (int len = 1; static_cast<std::size_t>(len) < count; len <<= 1) {
        const int blocks = static_cast<int>((pairs + threads - 1) / threads);
        fwht_stage_kernel<<<blocks, threads, 0, stream>>>(d_values, count, len);
        check_cuda(cudaGetLastError(), "launch fwht stage kernel");
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(count));
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    scale_kernel<<<blocks, threads, 0, stream>>>(d_values, count, inv_sqrt_n);
    check_cuda(cudaGetLastError(), "launch fwht scale kernel");
}

// (README) GPU Column-wise 正規化 Hadamard 轉換: 對每個 column 分別執行 normalized FWHT。
void fwht_columns_normalized_device(float* d_values, int padded_rows, int cols, cudaStream_t stream) {
    const int threads = 256;
    const std::size_t pairs = static_cast<std::size_t>(padded_rows) * cols / 2;
    for (int len = 1; len < padded_rows; len <<= 1) {
        const int blocks = static_cast<int>((pairs + threads - 1) / threads);
        fwht_columns_stage_kernel<<<blocks, threads, 0, stream>>>(d_values, padded_rows, cols, len);
        check_cuda(cudaGetLastError(), "launch column fwht stage kernel");
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(padded_rows));
    const int blocks = static_cast<int>(((static_cast<std::size_t>(padded_rows) * cols) + threads - 1) / threads);
    scale_kernel<<<blocks, threads, 0, stream>>>(
        d_values, static_cast<std::size_t>(padded_rows) * cols, inv_sqrt_n);
    check_cuda(cudaGetLastError(), "launch column fwht scale kernel");
}

void check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

std::size_t CompressedBlock::value_count() const {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

std::size_t CompressedBlock::payload_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    return codes.size() + sizeof(scale) + qjl_signs.size() + sizeof(residual_norm);
}

double CompressedBlock::compression_ratio_vs_fp32() const {
    const double fp32_bytes = static_cast<double>(value_count() * sizeof(float));
    if (fp32_bytes == 0.0) return 1.0;
    return fp32_bytes / static_cast<double>(std::max<std::size_t>(payload_bytes(), 1));
}

std::size_t DeviceCompressedBlock::value_count() const {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

std::size_t DeviceCompressedBlock::code_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    if (mode == QuantizeMode::kTurboQuant) {
        return lloyd_tq_code_bytes(rows, cols, bits);
    }
    const std::size_t count = static_cast<std::size_t>(padded_count);
    if (bits == 8) return count;
    if (bits == 4) return (count + 1) / 2;
    if (bits == 2) return (count + 3) / 4;
    return 0;
}

std::size_t DeviceCompressedBlock::payload_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    if (mode == QuantizeMode::kTurboQuant) {
        return code_bytes() + static_cast<std::size_t>(cols) * sizeof(float);
    }
    return code_bytes() + sizeof(scale) +
           ((mode == QuantizeMode::kTurboQuantQjl) ? static_cast<std::size_t>(qjl_dim) * sizeof(int) : 0) +
           sizeof(residual_norm);
}

// (README) 預設量化參數建立: 只用 bit 數建立 none 或 lowbit 模式的 QuantizeOptions。
QuantizeOptions make_quantize_options(int bits) {
    QuantizeOptions options;
    if (bits == 0) {
        options.mode = QuantizeMode::kNone;
        options.bits = 0;
        return options;
    }
    if (bits == 8 || bits == 4 || bits == 2) {
        options.mode = QuantizeMode::kLowBit;
        options.bits = bits;
        return options;
    }
    throw std::runtime_error("Unsupported --compress-b-bits value: " + std::to_string(bits));
}

// (README) CLI 量化參數解析: 將 mode、bit 數、QJL 參數與 seed 轉成 codec 會使用的 QuantizeOptions。
QuantizeOptions make_quantize_options(
    int bits,
    const std::string& mode,
    int qjl_dim,
    float qjl_alpha,
    unsigned seed) {
    QuantizeOptions options;
    options.bits = bits;
    options.qjl_dim = qjl_dim;
    options.qjl_alpha = qjl_alpha;
    options.seed = seed;

    if (mode == "none") {
        if (bits != 0) {
            throw std::runtime_error("mode=none requires bits=0.");
        }
        options.mode = QuantizeMode::kNone;
        return options;
    }
    if (mode == "lowbit") {
        if (bits != 8 && bits != 4 && bits != 2) {
            throw std::runtime_error("mode=lowbit requires bits=8, bits=4, or bits=2.");
        }
        options.mode = QuantizeMode::kLowBit;
        return options;
    }
    if (mode == "tq") {
        if (!is_supported_lloyd_tq_bits(bits)) {
            throw std::runtime_error("mode=tq requires bits in [2, 8].");
        }
        options.mode = QuantizeMode::kTurboQuant;
        options.qjl_dim = 0;
        options.qjl_alpha = 0.0f;
        return options;
    }
    if (mode == "tq-qjl") {
        if (bits != 8 && bits != 4 && bits != 2) {
            throw std::runtime_error("mode=tq-qjl requires bits=8, bits=4, or bits=2.");
        }
        if (qjl_dim <= 0) {
            throw std::runtime_error("mode=tq-qjl requires qjl_dim > 0.");
        }
        options.mode = QuantizeMode::kTurboQuantQjl;
        return options;
    }
    throw std::runtime_error("Unsupported quantization mode: " + mode);
}

// (README) CPU 端 legacy codec: 在 host vector 上執行 lowbit 或舊 tq-qjl prototype；正式 mode=tq 改由 d=256 Lloyd-Max device column path 處理。
CompressedBlock quantize_fp32_block(
    const std::vector<float>& values,
    int rows,
    int cols,
    const QuantizeOptions& options) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (values.size() != count) {
        throw std::runtime_error("Input size does not match compressed block dimensions.");
    }

    CompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;

    if (options.mode == QuantizeMode::kNone) {
        block.bits = 0;
        block.scale = 1.0f;
        block.codes.resize(count * sizeof(float));
        std::memcpy(block.codes.data(), values.data(), block.codes.size());
        return block;
    }
    if (options.mode == QuantizeMode::kTurboQuant) {
        throw std::runtime_error(
            "mode=tq uses Lloyd-Max codebook quantization and is supported only by the device column path with d=256.");
    }

    block.qjl_dim = options.qjl_dim;
    block.qjl_alpha = options.qjl_alpha;
    block.seed = options.seed;

    std::vector<float> values_for_quant = values;
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        block.padded_count = next_power_of_two(static_cast<int>(count));
        values_for_quant.assign(static_cast<std::size_t>(block.padded_count), 0.0f);
        std::copy(values.begin(), values.end(), values_for_quant.begin());
        apply_random_sign(values_for_quant, block.seed);
        fwht_normalized(values_for_quant);
    } else {
        block.padded_count = static_cast<int>(count);
    }

    const int qmax = qmax_for_bits(options.bits);
    float max_abs = 0.0f;
    for (float x : values_for_quant) max_abs = std::max(max_abs, std::fabs(x));

    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    std::vector<std::uint8_t> unpacked_codes(values_for_quant.size());
    for (std::size_t i = 0; i < values_for_quant.size(); ++i) {
        float scaled = values_for_quant[i] / block.scale;
        int q = static_cast<int>(std::lrintf(scaled));
        q = std::max(-qmax, std::min(qmax, q));
        unpacked_codes[i] = encode_signed_to_unsigned(q, options.bits);
    }
    block.codes = pack_codes(unpacked_codes, options.bits);

    if (options.mode == QuantizeMode::kTurboQuantQjl) {
        std::vector<float> reconstructed_rot(values_for_quant.size());
        for (std::size_t i = 0; i < values_for_quant.size(); ++i) {
            int q = decode_unsigned_to_signed(unpacked_codes[i], options.bits);
            reconstructed_rot[i] = static_cast<float>(q) * block.scale;
        }
        fwht_normalized(reconstructed_rot);
        apply_random_sign(reconstructed_rot, block.seed);

        std::vector<float> residual(static_cast<std::size_t>(block.padded_count), 0.0f);
        long double residual_norm2 = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            residual[i] = values[i] - reconstructed_rot[i];
            residual_norm2 += static_cast<long double>(residual[i]) * residual[i];
        }
        block.residual_norm = std::sqrt(static_cast<float>(residual_norm2));
        block.qjl_signs.assign((static_cast<std::size_t>(options.qjl_dim) + 7) / 8, 0);
        for (int s = 0; s < options.qjl_dim; ++s) {
            float dot = 0.0f;
            for (std::size_t j = 0; j < count; ++j) {
                dot += rademacher(block.seed + 17U, static_cast<std::uint32_t>(s),
                                  static_cast<std::uint32_t>(j)) * residual[j];
            }
            pack_sign_bit(block.qjl_signs, s, dot >= 0.0f);
        }
    }
    return block;
}

// (README) GPU block lowbit 壓縮: 將 device 上的 FP32 block 直接量化成 host-side compressed payload。
CompressedBlock quantize_fp32_device_block(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    cudaStream_t stream) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    CompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;

    if (options.mode == QuantizeMode::kNone) {
        block.bits = 0;
        block.scale = 1.0f;
        block.codes.resize(count * sizeof(float));
        check_cuda(cudaMemcpyAsync(
                       block.codes.data(), d_values, block.codes.size(),
                       cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync FP32 block");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize FP32 block");
        return block;
    }
    if (options.mode != QuantizeMode::kLowBit) {
        throw std::runtime_error("Device-side quantization currently supports only none and lowbit modes.");
    }

    const int qmax = qmax_for_bits(options.bits);
    thrust::device_ptr<const float> begin(d_values);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize max_abs");

    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    const std::size_t code_bytes = (options.bits == 8) ? count :
                                   (options.bits == 4) ? (count + 1) / 2 :
                                   (count + 3) / 4;
    block.codes.resize(code_bytes);

    std::uint8_t* d_codes = nullptr;
    check_cuda(cudaMalloc(&d_codes, code_bytes), "cudaMalloc quantized codes");

    const int threads = 256;
    if (options.bits == 8) {
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_values, d_codes, count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (count + 1) / 2;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_values, d_codes, count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (count + 3) / 4;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_values, d_codes, count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch quantization kernel");
    check_cuda(cudaMemcpyAsync(
                   block.codes.data(), d_codes, code_bytes,
                   cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync quantized codes");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize quantized codes");
    cudaFree(d_codes);
    return block;
}

// (README) GPU block legacy codec 壓縮並重建: 對 flattened block 做 lowbit 或舊 tq-qjl 壓縮；正式 mode=tq 不走此 uniform max-abs path。
CompressedBlock quantize_dequant_fp32_device_block_to_device(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    float* d_reconstructed,
    cudaStream_t stream,
    bool copy_payload_to_host) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (!d_reconstructed) {
        throw std::runtime_error("Device reconstructed output pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    if (options.mode != QuantizeMode::kNone &&
        options.mode != QuantizeMode::kLowBit &&
        options.mode != QuantizeMode::kTurboQuant &&
        options.mode != QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Unsupported device-side quantization mode.");
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

    CompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_dim : 0;
    block.qjl_alpha = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_alpha : 0.0f;
    block.seed = options.seed;

    if (options.mode == QuantizeMode::kNone) {
        block.bits = 0;
        block.scale = 1.0f;
        block.codes.resize(count * sizeof(float));
        if (copy_payload_to_host) {
            check_cuda(cudaMemcpyAsync(
                           block.codes.data(), d_values, block.codes.size(),
                           cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync FP32 block");
        }
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed, d_values, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync FP32 reconstructed device block");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize FP32 block");
        return block;
    }
    if (options.mode == QuantizeMode::kTurboQuant) {
        throw std::runtime_error(
            "mode=tq uses Lloyd-Max codebook quantization and is supported only by the device column path with d=256.");
    }

    qmax_for_bits(options.bits);
    const std::size_t work_count =
        (options.mode == QuantizeMode::kTurboQuant ||
         options.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(next_power_of_two(static_cast<int>(count))) :
        count;
    block.padded_count = static_cast<int>(work_count);

    float* d_work = nullptr;
    float* d_residual = nullptr;
    float* d_qjl_partials = nullptr;
    int* d_qjl_signs = nullptr;
    std::uint8_t* d_codes = nullptr;

    const std::size_t code_bytes = (options.bits == 8) ? work_count :
                                   (options.bits == 4) ? (work_count + 1) / 2 :
                                   (work_count + 3) / 4;
    block.codes.resize(code_bytes);

    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc quantization work");
    check_cuda(cudaMalloc(&d_codes, code_bytes), "cudaMalloc quantized codes");

    const int threads = 256;
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        copy_pad_random_sign_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_work, count, work_count, options.seed);
        check_cuda(cudaGetLastError(), "launch copy/pad/random-sign kernel");
        fwht_normalized_device(d_work, work_count, stream);
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_work, d_values, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit work block");
    }

    thrust::device_ptr<const float> begin(d_work);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + work_count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize max_abs");

    const int qmax = qmax_for_bits(options.bits);
    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    if (options.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (work_count + 1) / 2;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (work_count + 3) / 4;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch quantization kernel");

    if (options.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
    } else if (options.bits == 4) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
    } else if (options.bits == 2) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch dequantization kernel");

    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        fwht_normalized_device(d_work, work_count, stream);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        apply_random_sign_truncate_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_reconstructed, count, options.seed);
        check_cuda(cudaGetLastError(), "launch inverse random-sign/truncate kernel");
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed, d_work, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit reconstructed block");
    }

    if (options.mode == QuantizeMode::kTurboQuantQjl &&
        options.qjl_dim > 0 &&
        options.qjl_alpha != 0.0f) {
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        check_cuda(cudaMalloc(&d_residual, count * sizeof(float)), "cudaMalloc qjl residual");
        residual_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_reconstructed, d_residual, count);
        check_cuda(cudaGetLastError(), "launch qjl residual kernel");

        thrust::device_ptr<const float> residual_begin(d_residual);
        float residual_norm2 = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            residual_begin, residual_begin + count,
            SquareValue{},
            0.0f,
            thrust::plus<float>{});
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize qjl residual norm");
        block.residual_norm = std::sqrt(std::max(residual_norm2, 0.0f));

        const int qjl_threads = 256;
        const int blocks_per_sketch = 256;
        const std::size_t partial_count =
            static_cast<std::size_t>(options.qjl_dim) * blocks_per_sketch;
        check_cuda(cudaMalloc(&d_qjl_partials, partial_count * sizeof(float)),
                   "cudaMalloc qjl partials");
        check_cuda(cudaMalloc(&d_qjl_signs, static_cast<std::size_t>(options.qjl_dim) * sizeof(int)),
                   "cudaMalloc qjl signs");
        dim3 partial_grid(options.qjl_dim, blocks_per_sketch);
        qjl_dot_partial_kernel<<<partial_grid, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_residual, d_qjl_partials, count, options.qjl_dim, blocks_per_sketch,
            options.seed + 17U);
        check_cuda(cudaGetLastError(), "launch qjl dot partial kernel");
        qjl_pack_signs_kernel<<<options.qjl_dim, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_qjl_partials, d_qjl_signs, options.qjl_dim, blocks_per_sketch);
        check_cuda(cudaGetLastError(), "launch qjl pack signs kernel");

        const float coeff =
            options.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
            static_cast<float>(options.qjl_dim);
        qjl_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
            d_reconstructed, d_qjl_signs, count, options.qjl_dim, coeff, options.seed + 17U);
        check_cuda(cudaGetLastError(), "launch qjl reconstruction kernel");

        block.qjl_signs.assign((static_cast<std::size_t>(options.qjl_dim) + 7) / 8, 0);
        if (copy_payload_to_host) {
            std::vector<int> h_signs(options.qjl_dim, 0);
            check_cuda(cudaMemcpyAsync(
                           h_signs.data(), d_qjl_signs,
                           static_cast<std::size_t>(options.qjl_dim) * sizeof(int),
                           cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync qjl signs");
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize qjl signs");
            for (int s = 0; s < options.qjl_dim; ++s) {
                pack_sign_bit(block.qjl_signs, s, h_signs[s] != 0);
            }
        }
    } else if (options.mode == QuantizeMode::kTurboQuantQjl) {
        block.residual_norm = 0.0f;
        block.qjl_signs.assign((static_cast<std::size_t>(options.qjl_dim) + 7) / 8, 0);
    }

    if (copy_payload_to_host) {
        check_cuda(cudaMemcpyAsync(
                       block.codes.data(), d_codes, code_bytes,
                       cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync quantized codes");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize reconstructed quantized block");

    if (d_qjl_signs) cudaFree(d_qjl_signs);
    if (d_qjl_partials) cudaFree(d_qjl_partials);
    if (d_residual) cudaFree(d_residual);
    cudaFree(d_codes);
    cudaFree(d_work);
    return block;
}

// (README) Device payload 大小估算: 根據矩陣大小、padding 與 bit 數計算 compressed code buffer 需要的 bytes。
std::size_t device_code_bytes(int rows, int cols, const QuantizeOptions& options) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (options.mode == QuantizeMode::kNone) return count * sizeof(float);
    if (options.mode == QuantizeMode::kTurboQuant) {
        return lloyd_tq_code_bytes(rows, cols, options.bits);
    }
    qmax_for_bits(options.bits);
    const std::size_t work_count =
        (options.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(next_power_of_two(static_cast<int>(count))) :
        count;
    if (options.bits == 8) return work_count;
    if (options.bits == 4) return (work_count + 1) / 2;
    if (options.bits == 2) return (work_count + 3) / 4;
    return 0;
}

// (README) Column-wise TQ 符號預產生: 預先建立每個 column/padded row 的 D 符號矩陣以便重複使用。
void initialize_column_tq_signs(
    int rows,
    int cols,
    unsigned seed,
    signed char* d_signs,
    cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Column sign dimensions must be positive.");
    }
    if (!d_signs) {
        throw std::runtime_error("Column sign output pointer is null.");
    }
    const int padded_rows = next_power_of_two(rows);
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    initialize_column_signs_kernel<<<blocks, threads, 0, stream>>>(
        d_signs, padded_rows, cols, seed);
    check_cuda(cudaGetLastError(), "launch initialize column signs kernel");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize initialize column signs");
}

// (README) GPU block legacy payload 壓縮: 將 flattened device block 做 lowbit 或舊 tq-qjl payload；正式 mode=tq 不走此 uniform max-abs path。
DeviceCompressedBlock quantize_fp32_device_block_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    int* d_qjl_signs,
    cudaStream_t stream) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (!d_codes) {
        throw std::runtime_error("Device code output pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    if (options.mode == QuantizeMode::kNone) {
        throw std::runtime_error("Device payload path does not handle mode=none.");
    }
    if (options.mode == QuantizeMode::kTurboQuant) {
        throw std::runtime_error(
            "mode=tq uses Lloyd-Max codebook quantization and is supported only by the device column path with d=256.");
    }
    if (options.mode == QuantizeMode::kTurboQuantQjl && !d_qjl_signs) {
        throw std::runtime_error("tq-qjl device payload requires qjl sign storage.");
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    qmax_for_bits(options.bits);
    const std::size_t work_count =
        (options.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(next_power_of_two(static_cast<int>(count))) :
        count;

    DeviceCompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_dim : 0;
    block.qjl_alpha = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_alpha : 0.0f;
    block.seed = options.seed;
    block.padded_count = static_cast<int>(work_count);
    block.d_codes = d_codes;
    block.d_qjl_signs = d_qjl_signs;

    float* d_work = nullptr;
    float* d_reconstructed = nullptr;
    float* d_residual = nullptr;
    float* d_qjl_partials = nullptr;
    const std::size_t code_bytes = block.code_bytes();

    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc quantization work");

    const int threads = 256;
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        copy_pad_random_sign_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_work, count, work_count, options.seed);
        check_cuda(cudaGetLastError(), "launch copy/pad/random-sign kernel");
        fwht_normalized_device(d_work, work_count, stream);
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_work, d_values, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit work block");
    }

    thrust::device_ptr<const float> begin(d_work);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + work_count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize max_abs");

    const int qmax = qmax_for_bits(options.bits);
    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    if (options.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (work_count + 1) / 2;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (work_count + 3) / 4;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch quantization kernel");

    if (options.mode == QuantizeMode::kTurboQuantQjl &&
        options.qjl_dim > 0 &&
        options.qjl_alpha != 0.0f) {
        if (options.bits == 8) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 4) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 2) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        }
        check_cuda(cudaGetLastError(), "launch qjl source dequantization kernel");

        if (options.mode == QuantizeMode::kTurboQuantQjl) {
            fwht_normalized_device(d_work, work_count, stream);
            check_cuda(cudaMalloc(&d_reconstructed, count * sizeof(float)), "cudaMalloc qjl reconstructed scratch");
            const int blocks = static_cast<int>((count + threads - 1) / threads);
            apply_random_sign_truncate_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_reconstructed, count, options.seed);
            check_cuda(cudaGetLastError(), "launch inverse random-sign/truncate kernel");

            check_cuda(cudaMalloc(&d_residual, count * sizeof(float)), "cudaMalloc qjl residual");
            residual_kernel<<<blocks, threads, 0, stream>>>(
                d_values, d_reconstructed, d_residual, count);
            check_cuda(cudaGetLastError(), "launch qjl residual kernel");

            thrust::device_ptr<const float> residual_begin(d_residual);
            float residual_norm2 = thrust::transform_reduce(
                thrust::cuda::par.on(stream),
                residual_begin, residual_begin + count,
                SquareValue{},
                0.0f,
                thrust::plus<float>{});
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize qjl residual norm");
            block.residual_norm = std::sqrt(std::max(residual_norm2, 0.0f));

            const int qjl_threads = 256;
            const int blocks_per_sketch = 256;
            const std::size_t partial_count =
                static_cast<std::size_t>(options.qjl_dim) * blocks_per_sketch;
            check_cuda(cudaMalloc(&d_qjl_partials, partial_count * sizeof(float)),
                       "cudaMalloc qjl partials");
            dim3 partial_grid(options.qjl_dim, blocks_per_sketch);
            qjl_dot_partial_kernel<<<partial_grid, qjl_threads, qjl_threads * sizeof(float), stream>>>(
                d_residual, d_qjl_partials, count, options.qjl_dim, blocks_per_sketch,
                options.seed + 17U);
            check_cuda(cudaGetLastError(), "launch qjl dot partial kernel");
            qjl_pack_signs_kernel<<<options.qjl_dim, qjl_threads, qjl_threads * sizeof(float), stream>>>(
                d_qjl_partials, d_qjl_signs, options.qjl_dim, blocks_per_sketch);
            check_cuda(cudaGetLastError(), "launch qjl pack signs kernel");
        }
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize device payload quantization");
    if (d_qjl_partials) cudaFree(d_qjl_partials);
    if (d_residual) cudaFree(d_residual);
    if (d_reconstructed) cudaFree(d_reconstructed);
    cudaFree(d_work);
    (void)code_bytes;
    return block;
}

// (README) Column-wise TQ payload 壓縮: mode=tq 對每個 256 維 column 做 RHT preconditioning、Lloyd-Max scalar quantization 與 norm rescaling metadata。
DeviceCompressedBlock quantize_fp32_device_column_tq_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_norms,
    float* d_external_work,
    const signed char* d_signs,
    int* d_qjl_signs,
    float* d_qjl_reconstructed_ext,
    float* d_qjl_residual_ext,
    float* d_qjl_partials_ext,
    cudaStream_t stream) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (!d_codes) {
        throw std::runtime_error("Device code output pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    if (options.mode != QuantizeMode::kTurboQuant &&
        options.mode != QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Column TQ payload path requires mode=tq or mode=tq-qjl.");
    }

    if (options.mode == QuantizeMode::kTurboQuant) {
        if (rows != kLloydTqDim) {
            throw std::runtime_error("mode=tq Lloyd-Max column path requires vector dimension d=256.");
        }
        if (!is_supported_lloyd_tq_bits(options.bits)) {
            throw std::runtime_error("mode=tq Lloyd-Max column path requires bits in [2, 8].");
        }
        if (!d_norms) {
            throw std::runtime_error("mode=tq Lloyd-Max column path requires per-vector norm storage.");
        }
        if (d_signs) {
            throw std::runtime_error("mode=tq Lloyd-Max column path owns its Rademacher signs; external signs are unsupported.");
        }

        DeviceCompressedBlock block;
        block.rows = rows;
        block.cols = cols;
        block.bits = options.bits;
        block.mode = options.mode;
        block.qjl_dim = 0;
        block.qjl_alpha = 0.0f;
        block.seed = options.seed;
        block.padded_count = rows * cols;
        block.scale = 1.0f;
        block.residual_norm = 0.0f;
        block.d_codes = d_codes;
        block.d_norms = d_norms;

        check_cuda(cudaMemsetAsync(d_codes, 0, block.code_bytes(), stream),
                   "cudaMemsetAsync Lloyd-Max TQ codes");
        column_tq_lloyd_forward_kernel<<<
            cols, kLloydTqDim, 2 * kLloydTqDim * sizeof(float), stream>>>(
            d_values,
            d_codes,
            d_norms,
            rows,
            cols,
            options.bits,
            options.seed,
            1.0e-12f);
        check_cuda(cudaGetLastError(), "launch Lloyd-Max column TQ forward kernel");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize Lloyd-Max column TQ quantization");
        return block;
    }

    qmax_for_bits(options.bits);

    const int padded_rows = next_power_of_two(rows);
    const std::size_t work_count = static_cast<std::size_t>(padded_rows) * cols;

    DeviceCompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_dim : 0;
    block.qjl_alpha = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_alpha : 0.0f;
    block.seed = options.seed;
    block.padded_count = static_cast<int>(work_count);
    block.d_codes = d_codes;
    block.d_qjl_signs = d_qjl_signs;

    float* d_work = d_external_work;
    if (!d_work) {
        check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc column TQ work");
    }

    const int threads = 256;
    int blocks = static_cast<int>((work_count + threads - 1) / threads);
    if (!d_signs && padded_rows <= 1024) {
        column_tq_forward_fwht_kernel<<<cols, padded_rows, padded_rows * sizeof(float), stream>>>(
            d_values, d_work, rows, cols, padded_rows, options.seed);
        check_cuda(cudaGetLastError(), "launch shared column TQ forward FWHT kernel");
    } else if (d_signs) {
        copy_pad_apply_sign_columns_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_signs, d_work, rows, cols, padded_rows);
        check_cuda(cudaGetLastError(), "launch column copy/pad/sign-mask kernel");
        fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
    } else {
        copy_pad_random_sign_columns_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_work, rows, cols, padded_rows, options.seed);
        check_cuda(cudaGetLastError(), "launch column copy/pad/random-sign kernel");
        fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
    }

    thrust::device_ptr<const float> begin(d_work);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + work_count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column max_abs");

    const int qmax = qmax_for_bits(options.bits);
    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    if (options.bits == 8) {
        blocks = static_cast<int>((work_count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (work_count + 1) / 2;
        blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (work_count + 3) / 4;
        blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch column TQ quantization kernel");

    float* d_reconstructed = nullptr;
    bool d_reconstructed_owned = false;
    float* d_residual = nullptr;
    bool d_residual_owned = false;
    float* d_qjl_partials = nullptr;
    bool d_qjl_partials_owned = false;
    if (options.mode == QuantizeMode::kTurboQuantQjl &&
        options.qjl_dim > 0 &&
        options.qjl_alpha != 0.0f) {
        if (!block.d_qjl_signs) {
            throw std::runtime_error("Column tq-qjl payload requires qjl sign storage.");
        }
        if (options.bits == 8) {
            blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 4) {
            blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 2) {
            blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        }
        check_cuda(cudaGetLastError(), "launch column tq-qjl source dequantization kernel");

        const std::size_t count = static_cast<std::size_t>(rows) * cols;
        // Use external pre-allocated buffer if provided; else allocate internally.
        if (d_qjl_reconstructed_ext) {
            d_reconstructed = d_qjl_reconstructed_ext;
        } else {
            check_cuda(cudaMalloc(&d_reconstructed, count * sizeof(float)),
                       "cudaMalloc column tq-qjl reconstructed scratch");
            d_reconstructed_owned = true;
        }
        if (!d_signs && padded_rows <= 1024) {
            column_tq_inverse_fwht_store_kernel<<<cols, padded_rows, padded_rows * sizeof(float), stream>>>(
                d_work, d_reconstructed, rows, cols, padded_rows, options.seed);
            check_cuda(cudaGetLastError(), "launch shared column tq-qjl inverse FWHT store kernel");
        } else if (d_signs) {
            check_cuda(cudaMemsetAsync(d_reconstructed, 0, count * sizeof(float), stream),
                       "cudaMemsetAsync column tq-qjl reconstructed");
            fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
            blocks = static_cast<int>((count + threads - 1) / threads);
            apply_sign_mask_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_signs, d_reconstructed, rows, cols, padded_rows);
            check_cuda(cudaGetLastError(), "launch column tq-qjl inverse sign-mask/truncate kernel");
        } else {
            check_cuda(cudaMemsetAsync(d_reconstructed, 0, count * sizeof(float), stream),
                       "cudaMemsetAsync column tq-qjl reconstructed");
            fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
            blocks = static_cast<int>((count + threads - 1) / threads);
            apply_random_sign_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_reconstructed, rows, cols, padded_rows, options.seed);
            check_cuda(cudaGetLastError(), "launch column tq-qjl inverse random-sign/truncate kernel");
        }

        // Use external pre-allocated buffer if provided; else allocate internally.
        if (d_qjl_residual_ext) {
            d_residual = d_qjl_residual_ext;
        } else {
            check_cuda(cudaMalloc(&d_residual, count * sizeof(float)),
                       "cudaMalloc column tq-qjl residual");
            d_residual_owned = true;
        }
        blocks = static_cast<int>((count + threads - 1) / threads);
        residual_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_reconstructed, d_residual, count);
        check_cuda(cudaGetLastError(), "launch column tq-qjl residual kernel");

        thrust::device_ptr<const float> residual_begin(d_residual);
        float residual_norm2 = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            residual_begin, residual_begin + count,
            SquareValue{},
            0.0f,
            thrust::plus<float>{});
        // Must sync here to bring residual_norm to host (needed in block metadata).
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column tq-qjl residual norm");
        block.residual_norm = std::sqrt(std::max(residual_norm2, 0.0f));

        const int qjl_threads = 256;
        const int blocks_per_sketch = kQjlColumnBlocksPerSketch;
        const std::size_t partial_count =
            static_cast<std::size_t>(options.qjl_dim) * blocks_per_sketch;
        // Use external pre-allocated buffer if provided; else allocate internally.
        if (d_qjl_partials_ext) {
            d_qjl_partials = d_qjl_partials_ext;
        } else {
            check_cuda(cudaMalloc(&d_qjl_partials, partial_count * sizeof(float)),
                       "cudaMalloc column tq-qjl partials");
            d_qjl_partials_owned = true;
        }
        dim3 partial_grid(options.qjl_dim, blocks_per_sketch);
        qjl_dot_partial_kernel<<<partial_grid, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_residual, d_qjl_partials, count, options.qjl_dim, blocks_per_sketch,
            options.seed + 17U);
        check_cuda(cudaGetLastError(), "launch column tq-qjl dot partial kernel");
        qjl_pack_signs_kernel<<<options.qjl_dim, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_qjl_partials, block.d_qjl_signs, options.qjl_dim, blocks_per_sketch);
        check_cuda(cudaGetLastError(), "launch column tq-qjl pack signs kernel");
    }

    // Sync required for QJL (ensures pack_signs kernel writes are visible) or when
    // the work buffer is internally owned (must finish before we free it below).
    if (options.mode == QuantizeMode::kTurboQuantQjl || !d_external_work) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column TQ quantization");
    }

    // Only free buffers that were allocated internally (not external pre-allocs).
    if (d_qjl_partials_owned) cudaFree(d_qjl_partials);
    if (d_residual_owned)     cudaFree(d_residual);
    if (d_reconstructed_owned) cudaFree(d_reconstructed);

    if (!d_external_work) cudaFree(d_work);
    return block;
}

// (README) Device payload 解碼重建: 將 lowbit/TQ payload 解回完整 FP32 reconstructed matrix。
void dequantize_device_payload_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_reconstructed,
    cudaStream_t stream) {
    if (!block.d_codes) {
        throw std::runtime_error("Device compressed block has null codes.");
    }
    if (!d_reconstructed) {
        throw std::runtime_error("Device reconstructed output pointer is null.");
    }
    if (block.mode == QuantizeMode::kNone) {
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed,
                       reinterpret_cast<const float*>(block.d_codes),
                       block.value_count() * sizeof(float),
                       cudaMemcpyDeviceToDevice,
                       stream),
                   "cudaMemcpyAsync none payload to reconstructed");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize none payload");
        return;
    }
    if (block.mode == QuantizeMode::kTurboQuant) {
        throw std::runtime_error(
            "mode=tq Lloyd-Max payload is column-wise with per-vector norms; use dequantize_column_tq_payload_add_to_fp32.");
    }

    const std::size_t count = block.value_count();
    const std::size_t work_count = static_cast<std::size_t>(block.padded_count);
    float* d_work = nullptr;
    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc payload decode work");

    const int threads = 256;
    if (block.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 4) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 2) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else {
        cudaFree(d_work);
        throw std::runtime_error("Unsupported device payload bit width.");
    }
    check_cuda(cudaGetLastError(), "launch payload dequantization kernel");

    if (block.mode == QuantizeMode::kTurboQuant ||
        block.mode == QuantizeMode::kTurboQuantQjl) {
        fwht_normalized_device(d_work, work_count, stream);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        apply_random_sign_truncate_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_reconstructed, count, block.seed);
        check_cuda(cudaGetLastError(), "launch payload inverse random-sign/truncate kernel");
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed, d_work, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit payload reconstructed block");
    }

    if (block.mode == QuantizeMode::kTurboQuantQjl &&
        block.qjl_dim > 0 &&
        block.qjl_alpha != 0.0f &&
        block.residual_norm > 0.0f) {
        if (!block.d_qjl_signs) {
            cudaFree(d_work);
            throw std::runtime_error("tq-qjl payload decode requires qjl signs.");
        }
        const float coeff =
            block.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
            static_cast<float>(block.qjl_dim);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        qjl_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
            d_reconstructed, block.d_qjl_signs, count, block.qjl_dim, coeff, block.seed + 17U);
        check_cuda(cudaGetLastError(), "launch payload qjl reconstruction kernel");
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize payload decode");
    cudaFree(d_work);
}

// (README) Flattened payload 解碼累加: 將 lowbit payload 解碼後直接加到 FP32 accumulator；正式 mode=tq 需走 column-wise norm-aware decoder。
void dequantize_device_payload_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    const signed char* d_signs,
    cudaStream_t stream) {
    (void)d_signs;
    if (!block.d_codes) {
        throw std::runtime_error("Device compressed block has null codes.");
    }
    if (!d_accumulator || !d_work) {
        throw std::runtime_error("Device payload add path has null output/work pointer.");
    }
    if (block.mode == QuantizeMode::kNone) {
        const std::size_t count = block.value_count();
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        add_plain_kernel<<<blocks, threads, 0, stream>>>(
            d_accumulator, reinterpret_cast<const float*>(block.d_codes), count);
        check_cuda(cudaGetLastError(), "launch none payload add kernel");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize none payload add");
        return;
    }
    if (block.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Fused flattened payload add currently supports lowbit only, not tq-qjl.");
    }
    if (block.mode == QuantizeMode::kTurboQuant) {
        throw std::runtime_error(
            "mode=tq Lloyd-Max payload is column-wise with per-vector norms; use dequantize_column_tq_payload_add_to_fp32.");
    }

    const std::size_t count = block.value_count();
    const std::size_t work_count = static_cast<std::size_t>(block.padded_count);
    const int threads = 256;
    if (block.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 4) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 2) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else {
        throw std::runtime_error("Unsupported device payload bit width.");
    }
    check_cuda(cudaGetLastError(), "launch payload add dequantization kernel");

    if (block.mode == QuantizeMode::kTurboQuant) {
        fwht_normalized_device(d_work, work_count, stream);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        apply_random_sign_truncate_add_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_accumulator, count, block.seed);
        check_cuda(cudaGetLastError(), "launch payload inverse random-sign/truncate add kernel");
    } else {
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        add_plain_kernel<<<blocks, threads, 0, stream>>>(d_accumulator, d_work, count);
        check_cuda(cudaGetLastError(), "launch lowbit payload add kernel");
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize payload add decode");
}

// (README) Column-wise TQ payload 解碼累加: mode=tq 將 Lloyd-Max centroid 反 RHT 並乘回 norm，mode=tq-qjl 維持舊 column uniform path。
void dequantize_column_tq_payload_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    const signed char* d_signs,
    cudaStream_t stream) {
    if (!block.d_codes) {
        throw std::runtime_error("Device compressed block has null codes.");
    }
    if (!d_accumulator || !d_work) {
        throw std::runtime_error("Column TQ payload add path has null output/work pointer.");
    }
    if (block.mode != QuantizeMode::kTurboQuant &&
        block.mode != QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Column TQ payload add path requires mode=tq or mode=tq-qjl.");
    }
    if (block.mode == QuantizeMode::kTurboQuant) {
        if (block.rows != kLloydTqDim) {
            throw std::runtime_error("mode=tq Lloyd-Max decode requires vector dimension d=256.");
        }
        if (!is_supported_lloyd_tq_bits(block.bits)) {
            throw std::runtime_error("mode=tq Lloyd-Max decode requires bits in [2, 8].");
        }
        if (!block.d_norms) {
            throw std::runtime_error("mode=tq Lloyd-Max decode requires per-vector norm storage.");
        }
        if (d_signs) {
            throw std::runtime_error("mode=tq Lloyd-Max decode owns its Rademacher signs; external signs are unsupported.");
        }
        column_tq_lloyd_inverse_add_kernel<<<
            block.cols, kLloydTqDim, kLloydTqDim * sizeof(float), stream>>>(
            block.d_codes,
            block.d_norms,
            d_accumulator,
            block.rows,
            block.cols,
            block.bits,
            block.seed);
        check_cuda(cudaGetLastError(), "launch Lloyd-Max column TQ inverse add kernel");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize Lloyd-Max column TQ payload add decode");
        return;
    }
    const int padded_rows = static_cast<int>(static_cast<std::size_t>(block.padded_count) / block.cols);
    const std::size_t work_count = static_cast<std::size_t>(block.padded_count);
    const int threads = 256;

    if (!d_signs && padded_rows <= 1024) {
        if (block.bits != 8 && block.bits != 4 && block.bits != 2) {
            throw std::runtime_error("Unsupported column TQ payload bit width.");
        }
        column_tq_dequant_inverse_fwht_add_kernel<<<
            block.cols, padded_rows, padded_rows * sizeof(float), stream>>>(
            block.d_codes,
            d_accumulator,
            block.rows,
            block.cols,
            padded_rows,
            block.bits,
            block.scale,
            block.seed);
        check_cuda(cudaGetLastError(), "launch fused column TQ dequant/inverse FWHT add kernel");
    } else {
        if (block.bits == 8) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
        } else if (block.bits == 4) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
        } else if (block.bits == 2) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
        } else {
            throw std::runtime_error("Unsupported column TQ payload bit width.");
        }
        check_cuda(cudaGetLastError(), "launch column TQ payload dequantization kernel");

        if (d_signs) {
            fwht_columns_normalized_device(d_work, padded_rows, block.cols, stream);
            const int blocks = static_cast<int>((block.value_count() + threads - 1) / threads);
            apply_sign_mask_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_signs, d_accumulator, block.rows, block.cols, padded_rows);
            check_cuda(cudaGetLastError(), "launch column TQ inverse sign-mask/truncate add kernel");
        } else {
            fwht_columns_normalized_device(d_work, padded_rows, block.cols, stream);
            const int blocks = static_cast<int>((block.value_count() + threads - 1) / threads);
            apply_random_sign_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_accumulator, block.rows, block.cols, padded_rows, block.seed);
            check_cuda(cudaGetLastError(), "launch column TQ inverse random-sign/truncate add kernel");
        }
    }
    if (block.mode == QuantizeMode::kTurboQuantQjl &&
        block.qjl_dim > 0 &&
        block.qjl_alpha != 0.0f &&
        block.residual_norm > 0.0f) {
        if (!block.d_qjl_signs) {
            throw std::runtime_error("Column tq-qjl payload decode requires qjl signs.");
        }
        const float coeff =
            block.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
            static_cast<float>(block.qjl_dim);
        const int blocks = static_cast<int>((block.value_count() + threads - 1) / threads);
        qjl_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
            d_accumulator, block.d_qjl_signs, block.value_count(), block.qjl_dim, coeff, block.seed + 17U);
        check_cuda(cudaGetLastError(), "launch column tq-qjl reconstruction add kernel");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column TQ payload add decode");
}

// (README) GPU 壓縮重建測試入口: 將 device FP32 block 壓縮後解碼回 host vector 以便檢查誤差。
CompressedBlock quantize_dequant_fp32_device_block(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::vector<float>* reconstructed,
    cudaStream_t stream) {
    if (!reconstructed) {
        throw std::runtime_error("Reconstructed output pointer is null.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    reconstructed->assign(count, 0.0f);

    float* d_reconstructed = nullptr;
    check_cuda(cudaMalloc(&d_reconstructed, count * sizeof(float)), "cudaMalloc reconstructed block");
    CompressedBlock block = quantize_dequant_fp32_device_block_to_device(
        d_values, rows, cols, options, d_reconstructed, stream);
    check_cuda(cudaMemcpyAsync(
                   reconstructed->data(), d_reconstructed, count * sizeof(float),
                   cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync reconstructed block");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize reconstructed block");
    cudaFree(d_reconstructed);
    return block;
}

// (README) CPU payload 解碼: 將 host-side compressed payload 解回 FP32 vector，主要用於 prototype 與驗證。
std::vector<float> dequantize_fp32_block(const CompressedBlock& block) {
    if (block.rows <= 0 || block.cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    const std::size_t count = block.value_count();
    std::vector<float> values(count);

    if (block.mode == QuantizeMode::kNone) {
        if (block.codes.size() != count * sizeof(float)) {
            throw std::runtime_error("Invalid uncompressed payload size.");
        }
        std::memcpy(values.data(), block.codes.data(), block.codes.size());
        return values;
    }
    if (block.mode == QuantizeMode::kTurboQuant) {
        throw std::runtime_error(
            "mode=tq Lloyd-Max host payload decode is not implemented; use the device column path with d=256.");
    }

    qmax_for_bits(block.bits);
    const std::size_t decode_count =
        (block.mode == QuantizeMode::kTurboQuant ||
         block.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(block.padded_count) : count;
    std::vector<float> decoded(decode_count, 0.0f);
    for (std::size_t i = 0; i < decode_count; ++i) {
        std::uint8_t code = unpack_code_at(block.codes, i, block.bits);
        int q = decode_unsigned_to_signed(code, block.bits);
        decoded[i] = static_cast<float>(q) * block.scale;
    }

    if (block.mode == QuantizeMode::kTurboQuant ||
        block.mode == QuantizeMode::kTurboQuantQjl) {
        fwht_normalized(decoded);
        apply_random_sign(decoded, block.seed);
        std::copy(decoded.begin(), decoded.begin() + count, values.begin());

        if (block.qjl_dim > 0 && !block.qjl_signs.empty() && block.residual_norm > 0.0f) {
            const float coeff =
                block.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
                static_cast<float>(block.qjl_dim);
            for (std::size_t j = 0; j < count; ++j) {
                float accum = 0.0f;
                for (int s = 0; s < block.qjl_dim; ++s) {
                    float sign = unpack_sign_bit(block.qjl_signs, s) ? 1.0f : -1.0f;
                    accum += sign * rademacher(block.seed + 17U, static_cast<std::uint32_t>(s),
                                               static_cast<std::uint32_t>(j));
                }
                values[j] += coeff * accum;
            }
        }
    } else {
        std::copy(decoded.begin(), decoded.end(), values.begin());
    }
    return values;
}

// (README) 量化模式名稱: 將 QuantizeMode enum 轉成 log 會印出的文字名稱。
std::string mode_name(QuantizeMode mode) {
    switch (mode) {
        case QuantizeMode::kNone: return "none";
        case QuantizeMode::kLowBit: return "lowbit";
        case QuantizeMode::kTurboQuant: return "tq";
        case QuantizeMode::kTurboQuantQjl: return "tq-qjl";
        default: return "unknown";
    }
}

}  // namespace turboquant
