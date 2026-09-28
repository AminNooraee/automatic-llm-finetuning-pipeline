#!/bin/sh
set -eu

dns_values=${1-}
shift
[ "$#" -gt 0 ] || { echo "Build DNS wrapper requires a command." >&2; exit 2; }
[ -n "$dns_values" ] || exec "$@"

output=
for value in $dns_values; do
    case "$value" in
        *[!0-9A-Fa-f:.]*) echo "Invalid build DNS resolver literal." >&2; exit 2 ;;
    esac
    output="${output}nameserver ${value}
"
done
[ -n "$output" ] || { echo "No build DNS resolver was supplied." >&2; exit 2; }
backup=$(mktemp /tmp/pipeline-resolv.XXXXXX)
cp /etc/resolv.conf "$backup"
restore_resolver() {
    cat "$backup" > /etc/resolv.conf
    rm -f "$backup"
}
trap restore_resolver EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s' "$output" > /etc/resolv.conf
"$@"
