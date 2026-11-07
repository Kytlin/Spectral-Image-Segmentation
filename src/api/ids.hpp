#pragma once

#include <string>

namespace seg::api {

// RFC 4122 version-4 UUID (122 random bits) in canonical 8-4-4-4-12 lowercase hex.
std::string new_uuid();

}  // namespace seg::api
