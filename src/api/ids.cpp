#include "api/ids.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>

namespace seg::api {

std::string new_uuid() {
    // One generator per thread, seeded from the OS, so no lock is needed.
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::array<std::uint8_t, 16> b{};
    const std::uint64_t hi = rng();
    const std::uint64_t lo = rng();
    for (std::size_t i = 0; i < 8; ++i) {
        b[i] = static_cast<std::uint8_t>(hi >> (8 * i));
        b[i + 8] = static_cast<std::uint8_t>(lo >> (8 * i));
    }
    b[6] = static_cast<std::uint8_t>((b[6] & 0x0F) | 0x40);  // version 4
    b[8] = static_cast<std::uint8_t>((b[8] & 0x3F) | 0x80);  // variant 10
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(36);
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out += '-';
        out += kHex[b[i] >> 4];
        out += kHex[b[i] & 0x0F];
    }
    return out;
}

}  // namespace seg::api
