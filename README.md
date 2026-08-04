# Lerobot_Sanhak_Complete

세탁 파이프라인 **전 과정** — 이동로봇 **LeKiwi** 가 세탁기에서 수건을 꺼내 전달하고,
**SO101 bimanual** 이 이어받아 수건을 갠다. 본체 2대를 잇는 통합 코드다.

```
                    [시연 본체 lerobot-H310M-H — 전 과정을 여기서 돌린다]

 LeKiwi(Jetson, USB 이더넷)                          SO101 bimanual (ttyACM0/1)
 approach → 문열기 → VLA 집기 → carry → 복귀주행  ──▶  csi-agent 수건개기
   → throw → 전달 모션                                  rollout_auto (ACT 0~3 + classifier)
                          (task3 가 exit 0 일 때만 다음 단계로)
```

**전 과정이 시연 본체 한 대에서 돈다.** SmolVLA 추론(RTX 3050), Jetson 과의 ZMQ, 상공 카메라,
SO101 두 팔이 전부 이 본체에 붙어 있다. `laundry_task4.py` 는 csi-agent 가 로컬에 있으면
자동으로 로컬 모드가 되어 SSH 없이 두 단계를 잇는다.

두 대로 나눠 돌리는 구성(개발용: 랩탑이 LeKiwi 를 맡고 SO101 본체는 SSH 로 트리거)도
`--csi-host user@host` 로 그대로 지원한다.

- LeKiwi 단독 회수 태스크(하이브리드 Jetson 오케스트레이터 버전): https://github.com/LSJ-0829/Lerobot_Sanhak_VLA
- SO101 수건개기: https://github.com/lhwdev/csi-agent (`clothing/`)

## 구성

```
scripts/skills/laundry_task4.py   ★ 오케스트레이터 — 두 본체를 잇는 진입점
scripts/skills/laundry_task3.py        LeKiwi 세탁물 회수 전 과정 (랩탑 ZMQ 단일 프로세스)
examples/lekiwi/find_overhead_cam.py   상공 USB 카메라 자동 탐색(번호 대신)
examples/lekiwi/lekiwi_pose.py         poses/ JSON → 정규화 액션, 포즈 이동
examples/lekiwi/grasp_clip.py          CLIP grasp probe (GraspGate)
examples/lekiwi/lekiwi_calibration.json
poses/ motions/ red_approach.json      자세·모션·빨간 손잡이 HSV 파라미터
grasp_probe_overhead.pt                상공캠 grasp 게이트(VLA 정지 판단)
grasp_probe_wrist.pt                   손목캠 grasp 확정(상공 오탐 거르기)
```

디렉터리 구조는 lerobot 저장소와 같게 맞춰 뒀다 — `lekiwi_pose.py` 가 `parents[2]/poses` 를,
`laundry_task3.py` 가 `parents[2]/motions` 를 참조하기 때문에 이 배치 그대로 두어야 한다.
lerobot 체크아웃 위에 덮어써도 되고, 이 저장소만 단독으로 두고 써도 된다.

### 포함하지 않은 것

| 항목 | 어디서 | 왜 |
|---|---|---|
| SmolVLA 가중치 | HF `HyeonseokE/smolvla_lekiwi_spin_cycle` (base `lerobot/smolvla_base`) | 865MB |
| 수건개기 정책·classifier | SO101 본체 `csi-agent/lhwdev/train/` | 본체 로컬 학습물 |
| lerobot 본체 | https://github.com/huggingface/lerobot | 상위 의존성 |
| grasp_data_* 데이터셋 | 로컬 | 용량 |

## 셋업 (시연 본체에 처음 올릴 때)

```bash
bash tools/setup_demo_machine.sh
```

1. 상공 카메라 udev ID `/dev/lerobot/overhead` 설치
2. `camera_0` 충돌 수정 (아래 참고)
3. `open_clip_torch` 설치 (CLIP grasp probe)
4. SmolVLA 체크포인트 내려받기 (HF, 약 865MB → `models/`)
5. 검증 (`--check`)

네트워크는 건드리지 않는다. Jetson USB 이더넷은 아래 "Jetson 연결" 참고.

## 실행

```bash
# 0) 로봇을 움직이지 않고 연결·장비만 점검 — 항상 이것부터
python scripts/skills/laundry_task4.py --check

# 1) 전체 실행. Enter 는 한 번뿐이고 그 앞에서 모든 로딩이 끝난다
#    (첫 실전은 --pause 로 단계 사이를 한 번 더 끊는 걸 권함)
python scripts/skills/laundry_task4.py --pause

# 2) LeKiwi 는 이미 끝났고 수건개기만
python scripts/skills/laundry_task4.py --skip-task3

# `--` 뒤 인자는 전부 laundry_task3.py 로 전달
python scripts/skills/laundry_task4.py -- --record --skip-approach
```

전제: Jetson `lekiwi_host` 가 떠 있을 것(파이프라인이 SSH 로 자동 기동 시도),
상공 USB 카메라가 연결돼 `/dev/lerobot/overhead` 가 잡혀 있을 것.

## Jetson 연결 (LeKiwi) — 정적 IP 로만 붙인다

Jetson 은 USB-C 이더넷 가젯으로 붙는다. Jetson=`192.168.55.1`, 이 머신=`192.168.55.100`.

```bash
sudo install -m 755 tools/lekiwi-usbnet.sh /usr/local/sbin/lekiwi-usbnet.sh
sudo install -m 644 tools/lekiwi-usbnet.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now lekiwi-usbnet.service
ping -c1 192.168.55.1
```

### ⚠️ DHCP(`nmcli device connect`)를 쓰면 안 된다

Jetson 이 DHCP 로 **default route 까지 내려주고**, NetworkManager 가 그걸 설치하면서
이 머신의 기본 게이트웨이를 뺏어간다 — 인터넷과 원격 접속이 통째로 끊긴다.
2026-08-04 에 실제로 겪었고, 본체 앞에서 직접 복구해야 했다.

정적 IP 만 부여하면 link-scope 라우트(`192.168.55.0/24`)만 생기고 default route 는
구조적으로 만들어질 수 없다. `lekiwi-usbnet.sh` 가 그 방식이고, 해당 인터페이스를
NetworkManager `unmanaged` 로 돌려 DHCP 가 끼어들 여지도 없앤다.

Jetson 은 USB 가젯을 두 개(`enx…` 2개) 노출하는데 **어느 쪽이 살아있는지는 그때그때 다르다**
(실측: 한 번은 `…afe`, 재연결 후엔 `…afc`). 스크립트가 둘 다 시도해 응답하는 쪽에 붙인다.

⚠️ 메인 이더넷(`enp2s0`, 이 본체의 인터넷·SSH 회선)은 어떤 경우에도 건드리지 말 것.

### lekiwi_host 확인은 반드시 포트로

```bash
ssh comnet02@192.168.55.1 'ss -ltn | grep -E ":5555|:5556"'
```

`pgrep -f lekiwi_host` 는 **SSH 명령줄 자체를 매칭**해서 죽었는데도 "실행중" 으로 나온다.

## 두 단계를 어떻게 잇는가

1. **preflight 를 로봇이 움직이기 전에 먼저 돌린다.** SSH·원격 경로·파이썬·체크포인트·장치 링크와
   로컬 카메라·Jetson 을 모두 확인한다. 수건을 전달해 놓고 원격이 안 떠서 태스크가 통째로
   날아가는 걸 막으려는 것이다. 실패하면 시작하지 않는다(`--force-csi`/`--force-local` 로만 강행).

2. **종료코드로 인계한다.** `laundry_task3.py` 는 성공/실패를 종료코드로 알린다.

   | code | 뜻 |
   |---|---|
   | 0 | 성공 |
   | 3 | approach 실패 | 
   | 4 | 문열기 실패 |
   | 5 | 사용자 abort |
   | 6 | throw 모션 실패 |
   | 7 | 전달 모션 실패 |
   | 8 | 상공 카메라 못 찾음 |
   | 130 | Ctrl+C |

   `0` 일 때만 수건개기를 시작한다. exit 0 은 상공 게이트 + 손목 probe 2단계로 집기가
   확정되고 전달까지 끝났다는 뜻이다.

3. **전달 모션.** `--handoff-motion <이름>`(`motions/<이름>.json`) 을 주면 마지막 후퇴 *뒤에*
   재생한다. 안 주면 생략한다.

4. **수건개기 시작 조건.** `clothing/` 을 CWD 로 두고 `scripts/rollout_auto.py --no_step0` 을 돌린다.
   무거운 로딩은 첫 Enter 앞에서 이미 끝나 있고(prewarm), 프로세스는 IDLE 에서 대기하다가
   **top 카메라 classifier 가 '펴져 있음'을 인식하면 접기(step1)부터** 시작한다.

   step0(펼치기)는 정책이 불완전해 기본으로 제외한다(`--no_step0`). 그러면 rollout 의
   `start_step` 이 2 가 되어 IDLE 탈출 후보가 class 2~3 으로 좁혀진다 — 뭉쳐 있는 상태(class 1)로는
   시작하지 않고, 펴진 상태가 잡힐 때까지 기다린다.

   | 조건 | 값 |
   |---|---|
   | 후보 클래스 | class 2~3 (`--no_step0` 기준. step0 포함 시 1~2) |
   | 최소 확률 | 최근 15프레임 평균 ≥ 0.20 |
   | 유지 시간 | 확신도 0.6 → 5.0s, 1.0 → 1.0s (선형보간) |
   | 팔 움직임 배수 | idle 자세에서 0.05 rad 이상 벗어나거나 움직이면 최대 3.5배 |

   ⚠️ `--no_step0` 이면 stall 복구 2단계에서 step0 로 되돌아가지 못하므로, 접기가 두 번 막히면
   rollout 이 **종료**된다(step0 가 있으면 펼치기로 복구를 시도한다).

   ⚠️ 수건이 **top 카메라 시야(접는 판) 안에** 떨어져야 한다. 판 밖이면 계속 IDLE 이다.

## 본체마다 다른 값 (⚠️ 옮길 때 반드시 확인)

| 항목 | 왜 다른가 | 대응 |
|---|---|---|
| 상공 카메라 `/dev/videoN` | USB 를 다시 꽂을 때마다, 본체마다 번호가 바뀐다 | udev 로 `/dev/lerobot/overhead` 고정(`tools/99-lekiwi-overhead.rules`). 없으면 `find_overhead_cam.py` 가 자동 탐색 |
| Jetson IP | 유선 `192.168.55.1` / 무선 `192.168.0.19` | `--jetson-ip` 또는 `--wireless` |
| SO101 장치 | csi-agent 는 udev(`99-lerobot.rules`)로 `/dev/lerobot/{follower_1,2,camera_0,1,2}` 를 쓴다 | preflight 가 확인 |
| conda | 랩탑 miniforge3 / 시연 본체 miniconda3 | `--csi-python`, `LEROBOT_PYTHON` |

```bash
python examples/lekiwi/find_overhead_cam.py          # 뭘 골랐는지 + 고정 경로
python examples/lekiwi/find_overhead_cam.py --list   # 후보 전부 나열
```

### ⚠️ camera_0 충돌 — 상공캠이 top 카메라 이름을 뺏는다

이 리그의 카메라 4대는 **전부 같은 USB ID** 를 보고한다(`0c45:6366`, `SerialNumber=SN0001`).
csi-agent 의 `99-lerobot.rules` 는 top 카메라(`camera_0`)를 모델 문자열로 매칭하면서
"its model string is unique" 라고 적어 뒀는데, 상공캠을 꽂는 순간 그 전제가 깨진다 —
상공캠도 같은 Sonix / 같은 SN0001 이라 `/dev/lerobot/camera_0` 을 가져가 버리고,
수건개기 rollout 이 top 뷰 대신 세탁기를 보게 된다.

`tools/fix_camera0_conflict.sh` 가 `camera_0` 을 `camera_1`/`camera_2` 와 같은 방식
(물리 USB 포트 = `ID_PATH`)으로 바꿔 고친다. 원본은 `.bak` 으로 백업된다.

```
2026-08-04 실측 포트 배치
  camera_0  top          usb-0:7      (Sonix)
  camera_1  left arm     usb-0:3      (Innomaker)
  camera_2  right arm    usb-0:5      (Innomaker)
  overhead  세탁기 상공   usb-0:6.3    (Sonix)  ← 우리 것
```

카메라를 다른 포트에 옮겨 꽂으면 규칙의 `ID_PATH` 값을 바꿔야 한다.

## 주요 옵션

| 옵션 | 뜻 |
|---|---|
| `--check` | 로봇을 움직이지 않고 preflight 만 |
| `--wireless` | Jetson 무선(`192.168.0.19`). 없으면 유선 |
| `--pause` | task3 종료 후 Enter 를 한 번 더 받는다(기본은 Enter 한 번뿐) |
| `--csi-with-step0` | step0(펼치기)까지 포함(기본은 제외) |
| `--csi-immediate` | classifier 대기 없이 즉시 시작(`--no-prewarm` 과 함께) |
| `--no-prewarm` | 수건개기 모델을 미리 안 올림(=로딩 스톨 생김) |
| `--skip-task3` / `--skip-csi` | 한쪽 단계만 |
| `--record` | front/wrist/상공 프레임 녹화 → `runs/<ts>/` + mp4 |

세부 조정값(속도·시간·부호·임계값)은 전부 환경변수다 — `laundry_task3.py` 상단 CALIBRATION 섹션 참고.

## 원격 SO101 본체 실측값 (2026-08-04)

```
host    lerobot@115.145.179.95  (lerobot-H310M-H, Ubuntu 22.04, RTX 3050)
repo    /home/lerobot/lerobot2/csi-agent/lhwdev     (clothing/, train/)
python  /home/lerobot/miniconda3/envs/lerobot/bin/python   (3.12.13, torch 2.11 cu130)
ckpt    train/rollout_hil_towel_fold01_step{0..3}/checkpoints/last
        train/towel_fold01_nextlevel                (classifier)
통합코드 사본  /home/lerobot/lerobot2/csi-agent/LSJ/
```
