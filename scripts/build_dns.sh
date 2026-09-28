#!/bin/sh
# Strict resolver parsing shared by the launcher and its tests.

normalize_build_dns() {
    dns_input=$1
    dns_mode=${2:-strict}
    printf '%s\n' "$dns_input" | awk -v mode="$dns_mode" '
        function ipv4(value, fields, count, idx) {
            count = split(value, fields, ".")
            if (count != 4) return 0
            for (idx = 1; idx <= 4; idx++) {
                if (fields[idx] !~ /^[0-9]+$/) return 0
                if (length(fields[idx]) > 1 && substr(fields[idx], 1, 1) == "0") return 0
                if (fields[idx] + 0 > 255) return 0
            }
            return 1
        }
        function ipv6_side(value, fields, count, idx) {
            if (value == "") return 0
            count = split(value, fields, ":")
            for (idx = 1; idx <= count; idx++) {
                if (fields[idx] !~ /^[0-9a-f]{1,4}$/) return -1
            }
            return count
        }
        function ipv6(value, lower, first_double, left, right, left_count, right_count) {
            lower = tolower(value)
            if (lower !~ /:/ || lower ~ /[^0-9a-f:.]/) return 0
            if (lower ~ /\./) {
                if (lower !~ /:[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/) return 0
                match(lower, /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/)
                if (!ipv4(substr(lower, RSTART, RLENGTH))) return 0
                sub(/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/, "0:0", lower)
            }
            first_double = index(lower, "::")
            if (first_double) {
                if (index(substr(lower, first_double + 2), "::")) return 0
                left = substr(lower, 1, first_double - 1)
                right = substr(lower, first_double + 2)
                left_count = ipv6_side(left)
                right_count = ipv6_side(right)
                if (left_count < 0 || right_count < 0) return 0
                return left_count + right_count < 8
            }
            return ipv6_side(lower) == 8
        }
        function loopback(value, compact) {
            if (value == "0.0.0.0" || value == "::") return 1
            if (value ~ /^127\./) return 1
            if (value ~ /:/) {
                compact = tolower(value)
                gsub(/[0:]/, "", compact)
                if (compact == "") return 1
                if (tolower(value) ~ /^([0:]*:)1$/) return 1
                if (tolower(value) ~ /^([0:]*:)(ffff:)?127\./) return 1
                if (tolower(value) ~ /^([0:]*:)ffff:7f[0-9a-f][0-9a-f]:/) return 1
            }
            return 0
        }
        BEGIN { invalid = 0; output = "" }
        {
            gsub(/,/, " ")
            for (field = 1; field <= NF; field++) {
                value = tolower($field)
                valid = ipv4(value) || ipv6(value)
                usable = valid && !loopback(value)
                if (!usable) {
                    if (mode == "strict") invalid = 1
                    continue
                }
                if (!seen[value]++) output = output (output == "" ? "" : " ") value
            }
        }
        END {
            if (invalid || output == "") exit 2
            print output
        }
    '
}

discover_build_dns() {
    resolv_conf=${1:-/etc/resolv.conf}
    [ -r "$resolv_conf" ] || return 2
    resolver_values=$(awk '$1 == "nameserver" { print $2 }' "$resolv_conf")
    normalize_build_dns "$resolver_values" discovery
}
