# Lerobot_Sanhak_Complete

세탁 파이프라인 **전 과정** — 이동로봇 **LeKiwi** 가 세탁기에서 수건을 꺼내 전달하고,
**SO101 bimanual** 이 이어받아 수건을 갠다. 본체 2대를 잇는 통합 코드다.

```
[랩탑 GPU + Jetson/LeKiwi]                            [SO101 bimanual 본체]
 approach → 문열기 → VLA 집기 → carry → 복귀주행         csi-agent 수건개기
   → throw → 전달 모션                                    rollout_auto (ACT 0~3 + classifier)
        └────────────── ssh 트리거 ──────────────────────────────┘
              (task3 가 exit 0 일 때만)
```

- LeKiwi 단독 회수 태스크(하이브리드 Jetson 오케스트레이터 버전): https://github.com/LSJ-0829/Lerobot_Sanhak_VLA
- SO101 수건개기: https://github.com/lhwdev/csi-agent (`clothing/`)

## 구성

```
scripts/skills/laundry_pipeline.py   ★ 오케스트레이터 — 두 본체를 잇는 진입점
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

## 실행

**오케스트레이터는 랩탑에서 돈다.** SmolVLA 추론(랩탑 GPU), Jetson 과의 ZMQ, 상공 카메라가
전부 랩탑에 붙어 있기 때문이다. SO101 쪽은 SSH 로 트리거만 한다.

```bash
# 0) 로봇을 움직이지 않고 연결·장비만 점검 — 항상 이것부터
python scripts/skills/laundry_pipeline.py --check

# 1) 전체 실행 (첫 실전은 --pause 로 단계 사이를 끊고, --csi-idle 로 SO101 을 대기시키는 걸 권함)
python scripts/skills/laundry_pipeline.py --wireless --csi-idle --pause

# 2) LeKiwi 는 이미 끝났고 SO101 만 트리거
python scripts/skills/laundry_pipeline.py --skip-task3

# `--` 뒤 인자는 전부 laundry_task3.py 로 전달
python scripts/skills/laundry_pipeline.py -- --wireless --record --skip-approach
```

전제: Jetson `lekiwi_host` 가 떠 있을 것(파이프라인이 SSH 로 자동 기동 시도),
상공 USB 카메라가 **랩탑에** 연결돼 있을 것.

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

4. **원격 트리거.** `clothing/` 을 CWD 로 두고 `scripts/rollout_auto.py --immediate_start` 를
   실행한다. task3 가 exit 0 이면 수건이 이미 전달된 상태이므로 IDLE 을 건너뛴다.
   `--csi-idle` 을 주면 classifier 가 수건을 인식할 때까지 기다린다(팔이 안 움직여 더 안전).

## 본체마다 다른 값 (⚠️ 옮길 때 반드시 확인)

| 항목 | 왜 다른가 | 대응 |
|---|---|---|
| 상공 카메라 `/dev/videoN` | USB 를 다시 꽂을 때마다, 본체마다 번호가 바뀐다. 랩탑은 내장 Intel IPU6 가 `video0~31` 을 전부 차지해 USB 캠은 남는 번호에 붙는다 | 기본 `auto` — `find_overhead_cam.py` 가 내장 카메라를 빼고 실제로 프레임이 나오는 USB 노드를 고른다. 여러 개면 `--overhead-match C920` |
| Jetson IP | 유선 `192.168.55.1` / 무선 `192.168.0.19` | `--jetson-ip` 또는 `--wireless` |
| SO101 장치 | csi-agent 는 udev(`99-lerobot.rules`)로 `/dev/lerobot/{follower_1,2,camera_0,1,2}` 심볼릭 링크를 쓴다 — LeKiwi 쪽 규칙과 다르다 | preflight 가 확인 |
| conda | 랩탑 miniforge3 / SO101 본체 miniconda3 | `--csi-python`, `LEROBOT_PYTHON` |

```bash
python examples/lekiwi/find_overhead_cam.py          # 뭘 골랐는지 + 고정 경로(/dev/v4l/by-id/...)
python examples/lekiwi/find_overhead_cam.py --list   # 후보 전부 나열
```

## 주요 옵션

| 옵션 | 뜻 |
|---|---|
| `--check` | 로봇을 움직이지 않고 preflight 만 |
| `--wireless` | Jetson 무선(`192.168.0.19`). 없으면 유선 |
| `--pause` | task3 종료 후 Enter 를 눌러야 SO101 시작 |
| `--csi-idle` | SO101 을 IDLE 에서 classifier 판단까지 대기 |
| `--csi-no-step0` | step0(펼치기) 없이 step1 부터 |
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
