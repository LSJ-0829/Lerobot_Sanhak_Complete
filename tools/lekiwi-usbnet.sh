#!/bin/bash
# Jetson(LeKiwi) USB 이더넷 링크를 올린다.  Jetson=192.168.55.1 / 이 머신=192.168.55.100
#
# 왜 DHCP 를 안 쓰는가:
#   nmcli 로 DHCP 를 돌리면 Jetson 이 내려주는 default route 를 NetworkManager 가 설치해서
#   이 머신의 기본 게이트웨이를 뺏어간다(= 원격 접속·인터넷이 끊긴다. 2026-08-04 실제로 겪음).
#   정적 IP 만 부여하면 link-scope 라우트(192.168.55.0/24)만 생기고 default route 는
#   구조적으로 만들어질 수 없다. enp2s0(본체 인터넷 회선)은 어떤 경우에도 건드리지 않는다.
#
# Jetson 은 USB 가젯을 두 개(RNDIS/CDC) 노출하므로, 실제로 응답하는 쪽을 찾아 붙인다.

set -u
JETSON_IP=192.168.55.1
HOST_IP=192.168.55.100/24

log() { echo "[lekiwi-usbnet] $*"; }

ifaces=$(ls /sys/class/net | grep '^enx' || true)
if [ -z "$ifaces" ]; then
  log "USB 이더넷 인터페이스가 없다 — Jetson USB 케이블/전원 확인"
  exit 1
fi

# NetworkManager 가 DHCP 로 끼어들지 못하게 한다(이 인터페이스들만).
for d in $ifaces; do
  nmcli device set "$d" managed no 2>/dev/null
done

for d in $ifaces; do
  ip link set "$d" up 2>/dev/null
  ip addr add "$HOST_IP" dev "$d" 2>/dev/null
  sleep 2
  if ping -c2 -W2 "$JETSON_IP" >/dev/null 2>&1; then
    log "연결됨: $d → $JETSON_IP"
    ip route | grep -q '^default via .* dev enp2s0' \
      && log "default route 정상(enp2s0 유지)" \
      || log "⚠️ default route 가 enp2s0 이 아니다 — 확인 필요"
    exit 0
  fi
  # 이 인터페이스가 아니었다 → 원복하고 다음 것 시도
  ip addr del "$HOST_IP" dev "$d" 2>/dev/null
  ip link set "$d" down 2>/dev/null
done

log "어느 인터페이스로도 $JETSON_IP 에 닿지 않는다 — Jetson 전원/USB 가젯 확인"
exit 1
