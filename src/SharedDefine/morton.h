#pragma once

#include "bits_utils.h"
#include "float_n.h"

struct Morton{
	uint64 data;

private:
    inline uint64 expand_bits(uint64 bits){
        bits = (bits | (bits << 32)) & 0xFFFF00000000FFFF;
		bits = (bits | (bits << 16)) & 0x00FF0000FF0000FF;
		bits = (bits | (bits <<  8)) & 0xF00F00F00F00F00F;
		bits = (bits | (bits <<  4)) & 0x30C30C30C30C30C3;
		return (bits | (bits <<  2)) & 0x9249249249249249;
    }

public:
	Morton(){
		data = 0xFFFFFFFFFFFFFFFF;
	}

    Morton(Float3 pos){

		const uint precision = 21;

		float x = clamp_scalar(pos[0] * (1ll << precision), 0.0f, (1ll << precision) - 1.0f);
		float y = clamp_scalar(pos[1] * (1ll << precision), 0.0f, (1ll << precision) - 1.0f);
		float z = clamp_scalar(pos[2] * (1ll << precision), 0.0f, (1ll << precision) - 1.0f);

		// uint64 xx = expand_bits(static_cast<uint64>(x)) << (66 - 3 * precision); // << 2
		// uint64 yy = expand_bits(static_cast<uint64>(y)) << (65 - 3 * precision); // << 1
		// uint64 zz = expand_bits(static_cast<uint64>(z)) << (64 - 3 * precision); // << 0

		uint64 xx = expand_bits(static_cast<uint64>(x)); // << 2
		uint64 yy = expand_bits(static_cast<uint64>(y)); // << 1
		uint64 zz = expand_bits(static_cast<uint64>(z)); // << 0

		data = (xx << 2) | (yy << 1) | (zz << 0);

    }

	inline bool operator<(CREF(Morton) right){
		return data < right.data;
	}

	inline bool operator>(CREF(Morton) right){
		return data > right.data;
	}

	inline bool operator<=(CREF(Morton) right){
		return data <= right.data;
	}

	inline bool operator==(CREF(Morton) right){
		return data == right.data;
	}

	inline friend bool operator==(CREF(Morton) left, CREF(Morton) right){
		return left.data == right.data;
	}

	inline uint64 operator^(CREF(Morton) right){
		return data ^ right.data;
	}

	inline friend uint64 operator^(CREF(Morton) left, CREF(Morton) right){
		return left.data ^ right.data;
	}

	// friend std::ostream& operator<<(std::ostream& os, CREF(Morton) self) {
    //     os << " " << self.data;
    //     return os;
    // }

};
struct Morton32 {
	uint data;

public:
    static inline uint expand_bits(uint bits){
        bits = (bits | (bits << 16)) & 0x030000FF;
        bits = (bits | (bits << 8)) & 0x0300F00F;
        bits = (bits | (bits << 4)) & 0x030C30C3;
        return (bits | (bits << 2)) & 0x09249249;
    }

public:
	Morton32() {
        data = 0xFFFFFFFF;
    }

    Morton32(Float3 pos) {
        const uint precision = 10;

        float x = clamp_scalar(pos[0] * (1 << precision), 0.0f, (1 << precision) - 1.0f);
        float y = clamp_scalar(pos[1] * (1 << precision), 0.0f, (1 << precision) - 1.0f);
        float z = clamp_scalar(pos[2] * (1 << precision), 0.0f, (1 << precision) - 1.0f);

        uint xx = expand_bits(static_cast<uint>(x));
        uint yy = expand_bits(static_cast<uint>(y));
        uint zz = expand_bits(static_cast<uint>(z));

        data = (xx << 2) | (yy << 1) | zz;
    }

	inline bool operator<(CREF(Morton32) right){
		return data < right.data;
	}

	inline bool operator>(CREF(Morton32) right){
		return data > right.data;
	}

	inline bool operator<=(CREF(Morton32) right){
		return data <= right.data;
	}

	inline bool operator==(CREF(Morton32) right){
		return data == right.data;
	}

	inline friend bool operator==(CREF(Morton32) left, CREF(Morton32) right){
		return left.data == right.data;
	}

	inline uint64 operator^(CREF(Morton32) right){
		return data ^ right.data;
	}

	inline friend uint64 operator^(CREF(Morton32) left, CREF(Morton32) right){
		return left.data ^ right.data;
	}

};
