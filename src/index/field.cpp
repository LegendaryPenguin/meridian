#include "meridian/index.hpp"

namespace meridian {

const char* FieldName(Field f) {
    switch (f) {
        case Field::kTitle: return "title";
        case Field::kBody:  return "body";
        case Field::kTags:  return "tags";
        default:            return "unknown";
    }
}

}  // namespace meridian
