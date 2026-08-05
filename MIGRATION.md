# 새 본체로 옮기기 (다른 컴퓨터 · 같은 로봇)

대상: `csi@115.145.179.95`
기존: `lerobot@115.145.179.95` (같은 IP — 새 컴퓨터가 그 자리를 받았다)

로봇(SO101 양팔 · LeKiwi · 카메라 4대)은 그대로 옮겨 붙이므로, **로봇에 딸린 값은
다시 만들 필요가 없다.** 컴퓨터에 딸린 것만 새로 세운다.

---

## ⚠️ 먼저 — 옛 본체를 지우기 전에 반드시 빼둘 것

git 에 없어서 **지우면 되살릴 수 없는 것들**이다. 다른 무엇보다 먼저 한다.

```bash
# 새 본체에서 실행(옛 본체가 아직 살아 있을 때)
OLD=lerobot@<옛-본체-IP>

# 1. 캘리브레이션 — 팔 4개. 이 로봇 개체의 값이라 다시 뜨려면 사람이 팔을 잡고 돌려야 한다
rsync -a $OLD:~/.lerobot/calibration/ ~/.lerobot/calibration/

# 2. 학습된 모델 — train/ 는 git 에 없다 (~1.5GB)
rsync -a --info=progress2 $OLD:~/lerobot2/csi-agent/lhwdev/train/ ~/lerobot2/csi-agent/lhwdev/train/

# 3. SmolVLA (LeKiwi 잡기용, 각 865MB)
rsync -a --info=progress2 $OLD:~/lerobot/models/ ~/lerobot/models/

# 4. 홈 자세 · 접근 기준값 · 포즈 · 모션
rsync -a $OLD:~/lerobot2/csi-agent/lhwdev/clothing/idle_posture.json ~/
rsync -a $OLD:~/lerobot2/csi-agent/LSJ/{poses,motions,red_approach.json,tools} ~/backup_lsj/

# 5. udev 규칙 (모터 시리얼이 박혀 있다)
scp $OLD:/etc/udev/rules.d/99-lerobot.rules ~/backup_lsj/
```

빼둔 것이 맞는지 확인 — 이게 통과해야 옛 본체를 지워도 된다:
```bash
ls ~/.lerobot/calibration/*/            # 팔 4개 json
du -sh ~/lerobot2/csi-agent/lhwdev/train/   # ~1.5GB
ls ~/backup_lsj/99-lerobot.rules
```

---

## 1. 파이썬 환경

```bash
# miniconda3 설치 후
conda create -n lerobot python=3.12 -y && conda activate lerobot
git clone https://github.com/huggingface/lerobot ~/lerobot && cd ~/lerobot
pip install -e ".[all]"
```

`laundry_task4` 는 `~/miniconda3` · `~/miniforge3` · `~/anaconda3` 의 `lerobot` env 를
**스스로 찾는다**. 다른 곳이면 `CSI_PYTHON` 으로 알려주면 된다.

## 2. 코드

```bash
mkdir -p ~/lerobot2 && cd ~/lerobot2
git clone https://github.com/lhwdev/csi-agent          # 폴딩(남의 레포)
cd csi-agent
git clone git@github.com:LSJ-0829/Lerobot_Sanhak_Complete.git LSJ   # 우리 코드
```

배치는 이 모양이어야 한다 — `rollout_auto.py` 가 `clothing/` 기준 `../train/…` 을 본다:
```
~/lerobot2/csi-agent/
  ├─ lhwdev/{clothing,train}/     ← 폴딩 코드 + 모델
  ├─ tools/                       ← udev 규칙, usb_reset
  └─ LSJ/                         ← 우리 코드 (task3/4, examples/lekiwi)
```

경로가 다르면 알려준다(두 변수 중 하나만 정하면 된다):
```bash
export CSI_ROOT=~/lerobot2/csi-agent      # 또는
export CSI_DIR=~/lerobot2/csi-agent/lhwdev
```

## 3. 장치 이름 (udev)

옛 본체에서 가져온 규칙을 그대로 넣는다. **모터 시리얼은 로봇에 딸린 값이라 그대로 맞다.**

```bash
sudo cp ~/backup_lsj/99-lerobot.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger --action=add
ls -l /dev/lerobot/          # follower_1,2 / leader_1,2 / camera_0,1,2 / overhead
```

⚠️ **카메라 규칙 2줄은 새 컴퓨터에서 틀린다.** `camera_1`/`camera_2` 는 물리 USB 포트
(`pci-0000:00:14.0-usb-0:3:1.0`)로 매칭하는데 그 경로가 메인보드마다 다르다.
고치지 않아도 된다 — `resolve_cameras.py` 가 매 실행마다 모델+화면내용으로 다시 배정한다.

⚠️ **`authorized` 0666 줄은 반드시 있어야 한다**(USB 전원 리셋이 sudo 없이 되는 근거):
```
SUBSYSTEM=="usb", ATTR{idVendor}=="0c45", ATTR{idProduct}=="6366", MODE="0666", GROUP="plugdev", RUN+="/bin/sh -c 'chmod 0666 /sys%p/authorized 2>/dev/null || true'"
```
확인: `python examples/lekiwi/usb_reset.py` 가 4대를 OK 로 보여야 한다.

계정을 `video`·`plugdev`·`dialout` 그룹에 넣는다(넣고 재로그인):
```bash
sudo usermod -aG video,plugdev,dialout,sudo csi
```

## 4. 되돌려 놓은 값들

```bash
cp ~/idle_posture.json ~/lerobot2/csi-agent/lhwdev/clothing/
cp -r ~/backup_lsj/{poses,motions,red_approach.json} ~/lerobot2/csi-agent/LSJ/
```

## 5. csi-agent 패치 2개 (남의 레포 — 새로 클론했으니 다시 걸어야 한다)

```bash
cd ~/lerobot2/csi-agent/LSJ
python examples/lekiwi/patch_csi_motor_retry.py    # 모터 재연결
python examples/lekiwi/patch_csi_camera_paths.py   # 카메라 경로
# 확인
python examples/lekiwi/patch_csi_motor_retry.py --check
python examples/lekiwi/patch_csi_camera_paths.py --check
```

## 6. Jetson 연결 (LeKiwi 쪽)

```bash
sudo cp ~/backup_lsj/tools/lekiwi-usbnet.sh /usr/local/bin/
sudo cp ~/backup_lsj/tools/lekiwi-usbnet.service /etc/systemd/system/
sudo systemctl enable --now lekiwi-usbnet.service
echo 'csi ALL=(root) NOPASSWD: /bin/systemctl restart lekiwi-usbnet.service, /bin/systemctl start lekiwi-usbnet.service' \
  | sudo tee /etc/sudoers.d/lekiwi-usbnet
ping -c1 192.168.55.1
```

⚠️ **정적 IP(`192.168.55.100/24`)로만 붙일 것.** `nmcli device connect` 로 DHCP 를 돌리면
Jetson 이 내려준 default route 가 본체의 기본 게이트웨이를 뺏어가 인터넷·SSH 가 끊긴다
(2026-08-04 실제 발생). Jetson 은 `enx…` 를 2개 노출하고 살아있는 쪽이 재연결마다 바뀌므로
둘 다 시도해야 한다 — 위 스크립트가 그 방식이다.

Jetson 쪽 `lekiwi.py` 패치(그립 P게인 32, 팔 속도 800/가속 40)는 **Jetson 안에 있으므로
본체를 바꿔도 그대로 살아 있다.** 확인만: `ssh comnet02@192.168.55.1 'python3 ~/lekiwi_patch.py --check'`

## 7. 확인 (로봇을 움직이지 않는다)

```bash
cd ~/lerobot2/csi-agent/LSJ
python examples/lekiwi/csi_paths.py          # 경로가 맞게 잡혔는지
python examples/lekiwi/usb_reset.py          # 카메라 4대
python examples/lekiwi/scan_models.py        # 모델 목록
python examples/lekiwi/verify_policies.py    # 정책 4개가 실제로 로드·추론되는지
python scripts/skills/laundry_task4.py --check
```

그 다음 실제 실행:
```bash
python scripts/skills/laundry_task4.py --skip-task3   # 폴딩만
python scripts/skills/laundry_task4.py                # 전체
```

---

## 새 컴퓨터에서 다시 봐야 하는 것

| 항목 | 왜 |
|---|---|
| 카메라 좌/우 배정 방향 | 카메라를 다시 달면 바뀔 수 있다. 엉뚱한 팔이 움직이면 `resolve_cameras.py --set-left-x-side left` |
| 상공캠 | `/dev/videoN` 번호는 본체마다 다르다. `--overhead-cam auto` 가 기본이라 대개 알아서 찾는다 |
| GPU | 옛 본체는 RTX 3050. 더 느리면 폴딩 30fps 가 안 나올 수 있다 |
| `red_approach.json` | 상공캠의 **설치 높이·각도**가 바뀌면 `CENTER_X 346 / TARGET_Y 17` 을 다시 재야 한다. 카메라를 그대로 옮겨 달면 그대로 맞다 |

## 컴퓨터가 바뀌어도 그대로인 것

- 캘리브레이션(로봇 개체값), udev 모터 시리얼, 카메라 VID:PID(`0c45:6366`)
- `idle_posture.json`, `poses/`, `motions/`
- Jetson 안의 모든 것(usbnet 서비스, `lekiwi.py` 패치, keepalive)

## 되돌리기

변경 이력과 원복 방법은 [`CHANGES_20260805.md`](./CHANGES_20260805.md) 참고.
