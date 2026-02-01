# 실전 프로젝트 (Practical Projects)

## 개요

지금까지 학습한 OpenCV 기술들을 종합하여 실제 응용 프로젝트를 구현합니다. 각 프로젝트는 여러 기술을 조합하여 완성된 애플리케이션을 만드는 과정을 단계별로 안내합니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 이전 모든 챕터의 내용

---

## 목차

1. [프로젝트 1: 문서 스캐너](#프로젝트-1-문서-스캐너)
2. [프로젝트 2: 차선 검출](#프로젝트-2-차선-검출)
3. [프로젝트 3: AR 마커 검출](#프로젝트-3-ar-마커-검출)
4. [프로젝트 4: 실시간 얼굴 필터](#프로젝트-4-실시간-얼굴-필터)
5. [프로젝트 5: 객체 추적 시스템](#프로젝트-5-객체-추적-시스템)
6. [연습 문제 및 확장 아이디어](#연습-문제-및-확장-아이디어)

---

## 프로젝트 1: 문서 스캐너

### 프로젝트 개요

```
문서 스캐너 (Document Scanner):
사진으로 찍은 문서를 정렬된 스캔 이미지로 변환

┌──────────────────┐        ┌──────────────────┐
│   촬영된 문서     │        │   스캔된 결과    │
│  /‾‾‾‾‾‾‾‾‾‾‾\   │        │ ┌──────────────┐ │
│ /             \  │  ──▶   │ │              │ │
│ \             /  │        │ │   문서 내용   │ │
│  \___________/   │        │ │              │ │
│                  │        │ └──────────────┘ │
└──────────────────┘        └──────────────────┘
     기울어진 원본                정렬된 결과

사용 기술:
- 엣지 검출 (Canny)
- 윤곽선 검출 (findContours)
- 다각형 근사 (approxPolyDP)
- 원근 변환 (warpPerspective)
- 이진화 (adaptiveThreshold)
```

### 단계별 구현

```python
import cv2
import numpy as np

class DocumentScanner:
    """문서 스캐너"""

    def __init__(self):
        pass

    def order_points(self, pts):
        """4개의 점을 순서대로 정렬 (좌상, 우상, 우하, 좌하)"""
        rect = np.zeros((4, 2), dtype=np.float32)

        # 좌상: x+y 합이 가장 작음
        # 우하: x+y 합이 가장 큼
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # 우상: y-x 차이가 가장 작음
        # 좌하: y-x 차이가 가장 큼
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image, pts):
        """원근 변환으로 문서 정렬"""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # 새 이미지 크기 계산
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # 목표 좌표
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # 원근 변환 행렬
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

    def find_document_contour(self, image):
        """문서 윤곽선 찾기"""
        # 전처리
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 엣지 검출
        edged = cv2.Canny(blur, 75, 200)

        # 모폴로지 연산으로 엣지 연결
        kernel = np.ones((5, 5), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        edged = cv2.erode(edged, kernel, iterations=1)

        # 윤곽선 검출
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # 가장 큰 4각형 윤곽선 찾기
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        document_contour = None
        for contour in contours[:5]:  # 상위 5개만 확인
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                document_contour = approx
                break

        return document_contour, edged

    def enhance_document(self, image):
        """문서 이미지 향상"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # 또는 OTSU 이진화
        # _, binary = cv2.threshold(gray, 0, 255,
        #                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def scan(self, image, enhance=True):
        """문서 스캔 전체 과정"""
        original = image.copy()
        height, width = image.shape[:2]

        # 처리를 위해 리사이즈 (비율 유지)
        ratio = 500.0 / height
        resized = cv2.resize(image, None, fx=ratio, fy=ratio)

        # 문서 윤곽선 찾기
        contour, edged = self.find_document_contour(resized)

        if contour is None:
            print("문서 윤곽선을 찾을 수 없습니다")
            return None, None

        # 원본 크기로 좌표 변환
        contour = contour.reshape(4, 2) / ratio

        # 원근 변환
        scanned = self.four_point_transform(original, contour)

        # 문서 향상 (선택사항)
        if enhance:
            scanned = self.enhance_document(scanned)

        return scanned, contour

    def visualize(self, image, contour):
        """결과 시각화"""
        vis = image.copy()
        if contour is not None:
            cv2.drawContours(vis, [contour.astype(int)], -1, (0, 255, 0), 3)

            # 코너 점 표시
            for point in contour:
                cv2.circle(vis, tuple(point.astype(int)), 10, (0, 0, 255), -1)

        return vis

# 사용 예
scanner = DocumentScanner()

# 이미지 로드
img = cv2.imread('document_photo.jpg')

# 스캔
scanned, contour = scanner.scan(img, enhance=True)

if scanned is not None:
    # 결과 시각화
    vis = scanner.visualize(img, contour)

    cv2.imshow('Original with Contour', vis)
    cv2.imshow('Scanned', scanned)
    cv2.waitKey(0)

    # 저장
    cv2.imwrite('scanned_document.jpg', scanned)
```

### 실시간 문서 스캐너

```python
import cv2
import numpy as np

def realtime_document_scanner():
    """실시간 문서 스캐너"""

    scanner = DocumentScanner()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 문서 윤곽선 검출
        height = frame.shape[0]
        ratio = 500.0 / height
        resized = cv2.resize(frame, None, fx=ratio, fy=ratio)

        contour, _ = scanner.find_document_contour(resized)

        display = frame.copy()

        if contour is not None:
            # 원본 크기로 변환
            contour = (contour.reshape(4, 2) / ratio).astype(int)

            # 윤곽선 그리기
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 3)

            # 안내 텍스트
            cv2.putText(display, "Press 's' to scan", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Document not detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Document Scanner', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and contour is not None:
            # 스캔 수행
            scanned, _ = scanner.scan(frame)
            if scanned is not None:
                cv2.imshow('Scanned', scanned)
                cv2.imwrite('scanned.jpg', scanned)

    cap.release()
    cv2.destroyAllWindows()

# 실행
# realtime_document_scanner()
```

---

## 프로젝트 2: 차선 검출

### 프로젝트 개요

```
차선 검출 (Lane Detection):
도로 영상에서 차선을 검출하고 시각화

┌────────────────────────────────────┐
│            도로 영상               │
│                                    │
│     ╲                    ╱         │
│      ╲      차선       ╱          │
│       ╲              ╱            │
│        ╲    검출   ╱              │
│         ╲        ╱                │
│          ╲      ╱                 │
│           ╲    ╱                  │
│            ╲  ╱                   │
└────────────────────────────────────┘

처리 파이프라인:
1. 관심 영역 (ROI) 설정
2. 색상 공간 변환 (HSV)
3. 흰색/노란색 마스크 생성
4. 캐니 엣지 검출
5. 허프 변환으로 직선 검출
6. 차선 합성
```

### 단계별 구현

```python
import cv2
import numpy as np

class LaneDetector:
    """차선 검출기"""

    def __init__(self):
        pass

    def region_of_interest(self, img):
        """관심 영역 마스킹 (도로 부분만)"""
        height, width = img.shape[:2]

        # 사다리꼴 ROI
        vertices = np.array([[
            (int(width * 0.1), height),           # 좌하
            (int(width * 0.4), int(height * 0.6)), # 좌상
            (int(width * 0.6), int(height * 0.6)), # 우상
            (int(width * 0.9), height)            # 우하
        ]], dtype=np.int32)

        mask = np.zeros_like(img)

        if len(img.shape) == 3:
            cv2.fillPoly(mask, vertices, (255, 255, 255))
        else:
            cv2.fillPoly(mask, vertices, 255)

        masked = cv2.bitwise_and(img, mask)
        return masked

    def color_filter(self, img):
        """색상 필터 (흰색/노란색 차선)"""
        # HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 흰색 마스크
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 노란색 마스크
        lower_yellow = np.array([15, 80, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 마스크 결합
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 마스크 적용
        filtered = cv2.bitwise_and(img, img, mask=combined_mask)

        return filtered, combined_mask

    def detect_edges(self, img):
        """엣지 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detect_lines(self, edges):
        """허프 변환으로 직선 검출"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,              # 거리 해상도 (픽셀)
            theta=np.pi/180,    # 각도 해상도 (라디안)
            threshold=50,       # 최소 투표 수
            minLineLength=50,   # 최소 선 길이
            maxLineGap=150      # 최대 간격
        )
        return lines

    def separate_lines(self, lines, img_width):
        """좌/우 차선 분리"""
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        center = img_width / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 기울기 계산
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # 기울기가 너무 작으면 무시 (수평선)
            if abs(slope) < 0.3:
                continue

            # 좌/우 분류
            if slope < 0 and x1 < center and x2 < center:
                left_lines.append(line[0])
            elif slope > 0 and x1 > center and x2 > center:
                right_lines.append(line[0])

        return left_lines, right_lines

    def average_line(self, lines, img_height):
        """여러 선분을 평균내어 하나의 선으로"""
        if len(lines) == 0:
            return None

        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        # 선형 회귀 (1차 다항식 피팅)
        poly = np.polyfit(y_coords, x_coords, deg=1)

        # y 범위 설정
        y1 = img_height
        y2 = int(img_height * 0.6)

        # x 좌표 계산
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))

        return [x1, y1, x2, y2]

    def draw_lanes(self, img, left_line, right_line):
        """차선 그리기"""
        overlay = np.zeros_like(img)

        # 차선 그리기
        if left_line is not None:
            cv2.line(overlay, (left_line[0], left_line[1]),
                    (left_line[2], left_line[3]), (0, 0, 255), 10)

        if right_line is not None:
            cv2.line(overlay, (right_line[0], right_line[1]),
                    (right_line[2], right_line[3]), (0, 0, 255), 10)

        # 차선 영역 채우기
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], np.int32)

            cv2.fillPoly(overlay, [pts], (0, 255, 0))

        # 원본과 합성
        result = cv2.addWeighted(img, 1, overlay, 0.3, 0)

        return result

    def detect(self, img):
        """전체 차선 검출 파이프라인"""
        height, width = img.shape[:2]

        # 1. 색상 필터링
        filtered, color_mask = self.color_filter(img)

        # 2. 엣지 검출
        edges = self.detect_edges(filtered)

        # 3. ROI 적용
        roi_edges = self.region_of_interest(edges)

        # 4. 직선 검출
        lines = self.detect_lines(roi_edges)

        # 5. 좌/우 차선 분리
        left_lines, right_lines = self.separate_lines(lines, width)

        # 6. 평균 차선 계산
        left_lane = self.average_line(left_lines, height)
        right_lane = self.average_line(right_lines, height)

        # 7. 결과 시각화
        result = self.draw_lanes(img, left_lane, right_lane)

        return result, {
            'edges': roi_edges,
            'color_mask': color_mask,
            'left_lane': left_lane,
            'right_lane': right_lane
        }

# 사용 예
detector = LaneDetector()

# 이미지에서 차선 검출
img = cv2.imread('road.jpg')
result, debug = detector.detect(img)

cv2.imshow('Lane Detection', result)
cv2.imshow('Edges', debug['edges'])
cv2.waitKey(0)
```

### 비디오 차선 검출

```python
import cv2
import numpy as np

def video_lane_detection(video_path):
    """비디오에서 차선 검출"""

    detector = LaneDetector()
    cap = cv2.VideoCapture(video_path)

    # 출력 비디오 설정
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('lane_output.mp4', fourcc, fps, (width, height))

    # 이전 프레임의 차선 (스무딩용)
    prev_left = None
    prev_right = None
    alpha = 0.7  # 스무딩 계수

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, debug = detector.detect(frame)

        # 차선 스무딩 (급격한 변화 방지)
        left = debug['left_lane']
        right = debug['right_lane']

        if prev_left is not None and left is not None:
            left = [int(alpha * prev_left[i] + (1 - alpha) * left[i])
                    for i in range(4)]
        if prev_right is not None and right is not None:
            right = [int(alpha * prev_right[i] + (1 - alpha) * right[i])
                     for i in range(4)]

        prev_left = left
        prev_right = right

        # 스무딩된 차선으로 다시 그리기
        result = detector.draw_lanes(frame, left, right)

        out.write(result)
        cv2.imshow('Lane Detection', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 실행
# video_lane_detection('driving.mp4')
```

---

## 프로젝트 3: AR 마커 검출

### 프로젝트 개요

```
AR 마커 검출 (AR Marker Detection):
이미지에서 정사각형 마커를 검출하고 3D 객체를 합성

마커 구조:
┌────────────────────┐
│ ██████████████████ │
│ █                █ │
│ █  ████    ████  █ │
│ █  ████    ████  █ │
│ █                █ │
│ █  ████████████  █ │
│ █  ████████████  █ │
│ █                █ │
│ ██████████████████ │
└────────────────────┘

처리 과정:
1. 사각형 윤곽선 검출
2. 원근 변환으로 마커 정규화
3. 마커 ID 인식
4. 호모그래피로 3D 객체 투영
```

### 단계별 구현

```python
import cv2
import numpy as np

class ARMarkerDetector:
    """AR 마커 검출기"""

    def __init__(self, marker_size=100):
        self.marker_size = marker_size

    def order_points(self, pts):
        """4개 점을 순서대로 정렬"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 좌상
        rect[2] = pts[np.argmax(s)]  # 우하

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 우상
        rect[3] = pts[np.argmax(diff)]  # 좌하

        return rect

    def find_markers(self, img):
        """마커 후보 찾기"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # 윤곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)

        markers = []

        for contour in contours:
            # 면적 필터
            area = cv2.contourArea(contour)
            if area < 1000 or area > img.shape[0] * img.shape[1] * 0.5:
                continue

            # 다각형 근사
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            # 4각형인 경우만
            if len(approx) == 4:
                # 볼록 다각형 확인
                if cv2.isContourConvex(approx):
                    markers.append(approx.reshape(4, 2))

        return markers, binary

    def get_marker_transform(self, corners):
        """마커 정규화를 위한 변환 행렬"""
        ordered = self.order_points(corners.astype(np.float32))

        dst = np.array([
            [0, 0],
            [self.marker_size - 1, 0],
            [self.marker_size - 1, self.marker_size - 1],
            [0, self.marker_size - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        return M, ordered

    def decode_marker(self, warped):
        """마커 ID 디코딩 (간단한 예)"""
        # 그레이스케일 변환
        if len(warped.shape) == 3:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # 이진화
        _, binary = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

        # 5x5 그리드로 분할 (가장자리는 검은 테두리)
        grid_size = self.marker_size // 5
        grid = np.zeros((5, 5), dtype=np.uint8)

        for i in range(5):
            for j in range(5):
                cell = binary[i*grid_size:(i+1)*grid_size,
                             j*grid_size:(j+1)*grid_size]
                # 셀의 평균 밝기로 0/1 결정
                grid[i, j] = 1 if np.mean(cell) > 127 else 0

        # 간단한 ID 계산 (내부 3x3 영역)
        inner = grid[1:4, 1:4]
        marker_id = 0
        for i in range(3):
            for j in range(3):
                marker_id = marker_id * 2 + inner[i, j]

        return marker_id, grid

    def draw_cube(self, img, corners, size=50):
        """마커 위에 3D 큐브 그리기"""
        # 마커 평면의 4개 점
        corners = self.order_points(corners.astype(np.float32))

        # 바닥면 좌표
        bottom = corners.astype(int)

        # 윗면 좌표 계산 (호모그래피 이용한 간단한 근사)
        center = np.mean(corners, axis=0)

        # 윗면은 마커 중심 방향으로 축소 + 위로 이동
        scale = 0.7
        offset = np.array([0, -size])  # 위로 이동

        top = []
        for pt in corners:
            vec = pt - center
            new_pt = center + vec * scale + offset
            top.append(new_pt.astype(int))
        top = np.array(top)

        # 면 그리기 (반투명)
        overlay = img.copy()

        # 윗면 (빨간색)
        cv2.fillPoly(overlay, [top], (0, 0, 200))

        # 옆면 (녹색)
        for i in range(4):
            pts = np.array([bottom[i], bottom[(i+1)%4],
                           top[(i+1)%4], top[i]])
            cv2.fillPoly(overlay, [pts], (0, 200, 0))

        # 합성
        result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        # 엣지 그리기
        for i in range(4):
            cv2.line(result, tuple(bottom[i]), tuple(bottom[(i+1)%4]),
                    (255, 255, 255), 2)
            cv2.line(result, tuple(top[i]), tuple(top[(i+1)%4]),
                    (255, 255, 255), 2)
            cv2.line(result, tuple(bottom[i]), tuple(top[i]),
                    (255, 255, 255), 2)

        return result

    def detect(self, img):
        """마커 검출 및 AR 렌더링"""
        result = img.copy()

        markers, binary = self.find_markers(img)

        detected_markers = []

        for corners in markers:
            # 마커 정규화
            M, ordered = self.get_marker_transform(corners)
            warped = cv2.warpPerspective(img, M,
                                         (self.marker_size, self.marker_size))

            # 마커 ID 디코딩
            marker_id, grid = self.decode_marker(warped)

            # 테두리 확인 (가장자리가 검은색이어야 함)
            border_check = (grid[0, :].sum() + grid[4, :].sum() +
                           grid[:, 0].sum() + grid[:, 4].sum())

            if border_check < 5:  # 대부분 검은색
                detected_markers.append({
                    'id': marker_id,
                    'corners': ordered
                })

                # 3D 큐브 그리기
                result = self.draw_cube(result, ordered)

                # ID 표시
                center = np.mean(ordered, axis=0).astype(int)
                cv2.putText(result, f"ID: {marker_id}", tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return result, detected_markers, binary

# 사용 예
detector = ARMarkerDetector()

# 이미지에서 마커 검출
img = cv2.imread('ar_marker.jpg')
result, markers, binary = detector.detect(img)

print(f"검출된 마커: {len(markers)}")
for m in markers:
    print(f"  ID: {m['id']}")

cv2.imshow('AR Detection', result)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
```

### ArUco 마커 사용 (OpenCV 내장)

```python
import cv2
import numpy as np

def aruco_marker_detection():
    """OpenCV ArUco 마커 검출"""

    # ArUco 딕셔너리 선택
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 마커 검출
        corners, ids, rejected = detector.detectMarkers(frame)

        # 결과 시각화
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                # 각 마커에 큐브 또는 축 그리기
                # (카메라 캘리브레이션이 있는 경우)
                pass

        cv2.imshow('ArUco Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def generate_aruco_marker(marker_id=0, size=200):
    """ArUco 마커 생성"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)

    cv2.imwrite(f'aruco_marker_{marker_id}.png', marker_img)
    return marker_img

# 마커 생성
# marker = generate_aruco_marker(0)
# cv2.imshow('Marker', marker)
```

---

## 프로젝트 4: 실시간 얼굴 필터

### 프로젝트 개요

```
실시간 얼굴 필터 (Face Filter):
얼굴 랜드마크를 기반으로 필터 효과 적용

┌────────────────────────────────────┐
│                                    │
│        😎 선글라스 필터            │
│       /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\           │
│      │  ●────────●   │           │
│      │   \      /    │           │
│       \    ▽       /            │
│        \   ∪     /              │
│                                    │
└────────────────────────────────────┘

사용 기술:
- dlib 얼굴 랜드마크 (68점)
- 투명 이미지 합성
- 어파인/원근 변환
- 실시간 처리 최적화
```

### 단계별 구현

```python
import cv2
import numpy as np
import dlib

class FaceFilter:
    """실시간 얼굴 필터"""

    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # 필터 이미지 로드
        self.filters = {}

    def load_filter(self, name, image_path, alpha_path=None):
        """필터 이미지 로드 (PNG with alpha 권장)"""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img.shape[2] == 4:
            # 이미 알파 채널 있음
            self.filters[name] = img
        else:
            # 알파 채널 추가 (흰색 배경을 투명으로)
            if alpha_path:
                alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(gray, 250, 255,
                                         cv2.THRESH_BINARY_INV)

            b, g, r = cv2.split(img)
            self.filters[name] = cv2.merge([b, g, r, alpha])

    def get_landmarks(self, img, face):
        """얼굴 랜드마크 추출"""
        shape = self.predictor(img, face)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                              for i in range(68)])
        return landmarks

    def overlay_image(self, background, overlay, x, y):
        """투명 이미지 합성"""
        h, w = overlay.shape[:2]

        # 경계 체크
        if x < 0:
            overlay = overlay[:, -x:]
            w = overlay.shape[1]
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            h = overlay.shape[0]
            y = 0

        bh, bw = background.shape[:2]
        if x + w > bw:
            overlay = overlay[:, :bw - x]
            w = overlay.shape[1]
        if y + h > bh:
            overlay = overlay[:bh - y, :]
            h = overlay.shape[0]

        if w <= 0 or h <= 0:
            return background

        # 알파 블렌딩
        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0

        roi = background[y:y+h, x:x+w]

        for c in range(3):
            roi[:, :, c] = (alpha * overlay_rgb[:, :, c] +
                           (1 - alpha) * roi[:, :, c])

        background[y:y+h, x:x+w] = roi

        return background

    def apply_sunglasses(self, img, landmarks, filter_img):
        """선글라스 필터 적용"""
        # 눈 좌표
        left_eye = landmarks[36:42].mean(axis=0).astype(int)
        right_eye = landmarks[42:48].mean(axis=0).astype(int)

        # 눈 사이 거리와 각도
        eye_width = np.linalg.norm(right_eye - left_eye)
        eye_center = ((left_eye + right_eye) / 2).astype(int)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                      right_eye[0] - left_eye[0]))

        # 선글라스 크기 조정
        filter_width = int(eye_width * 2.5)
        filter_height = int(filter_width * filter_img.shape[0] /
                           filter_img.shape[1])

        resized_filter = cv2.resize(filter_img, (filter_width, filter_height))

        # 회전
        M = cv2.getRotationMatrix2D((filter_width // 2, filter_height // 2),
                                    -angle, 1)
        rotated_filter = cv2.warpAffine(resized_filter, M,
                                        (filter_width, filter_height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0, 0))

        # 위치 계산
        x = eye_center[0] - filter_width // 2
        y = eye_center[1] - filter_height // 2

        # 합성
        result = self.overlay_image(img, rotated_filter, x, y)

        return result

    def apply_hat(self, img, landmarks, filter_img):
        """모자 필터 적용"""
        # 이마 위치 (눈썹 위)
        left_brow = landmarks[17:22].mean(axis=0)
        right_brow = landmarks[22:27].mean(axis=0)

        brow_center = ((left_brow + right_brow) / 2).astype(int)
        brow_width = np.linalg.norm(right_brow - left_brow)

        # 모자 크기
        hat_width = int(brow_width * 3)
        hat_height = int(hat_width * filter_img.shape[0] /
                        filter_img.shape[1])

        resized_hat = cv2.resize(filter_img, (hat_width, hat_height))

        # 위치 (눈썹 위에 배치)
        x = brow_center[0] - hat_width // 2
        y = brow_center[1] - hat_height

        # 합성
        result = self.overlay_image(img, resized_hat, x, y)

        return result

    def apply_mustache(self, img, landmarks, filter_img):
        """콧수염 필터 적용"""
        # 코 아래, 입 위
        nose_tip = landmarks[33]
        upper_lip = landmarks[51]

        center = ((nose_tip + upper_lip) / 2).astype(int)

        # 콧수염 크기 (입 너비 기준)
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        mustache_width = int(mouth_width * 1.5)
        mustache_height = int(mustache_width * filter_img.shape[0] /
                             filter_img.shape[1])

        resized = cv2.resize(filter_img, (mustache_width, mustache_height))

        x = center[0] - mustache_width // 2
        y = center[1] - mustache_height // 2

        result = self.overlay_image(img, resized, x, y)

        return result

    def process(self, img, filter_name='sunglasses'):
        """필터 적용"""
        if filter_name not in self.filters:
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 0)

        result = img.copy()

        for face in faces:
            landmarks = self.get_landmarks(rgb, face)

            if filter_name == 'sunglasses':
                result = self.apply_sunglasses(result, landmarks,
                                               self.filters[filter_name])
            elif filter_name == 'hat':
                result = self.apply_hat(result, landmarks,
                                        self.filters[filter_name])
            elif filter_name == 'mustache':
                result = self.apply_mustache(result, landmarks,
                                             self.filters[filter_name])

        return result

# 사용 예
def realtime_face_filter():
    """실시간 얼굴 필터"""

    filter_app = FaceFilter('shape_predictor_68_face_landmarks.dat')

    # 필터 로드 (투명 PNG 권장)
    filter_app.load_filter('sunglasses', 'sunglasses.png')
    # filter_app.load_filter('hat', 'hat.png')
    # filter_app.load_filter('mustache', 'mustache.png')

    cap = cv2.VideoCapture(0)

    current_filter = 'sunglasses'
    filters = list(filter_app.filters.keys())
    filter_idx = 0

    print("Press 'n' to change filter, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 필터 적용
        result = filter_app.process(frame, current_filter)

        # 현재 필터 표시
        cv2.putText(result, f"Filter: {current_filter}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Filter', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            filter_idx = (filter_idx + 1) % len(filters)
            current_filter = filters[filter_idx]

    cap.release()
    cv2.destroyAllWindows()

# 실행
# realtime_face_filter()
```

---

## 프로젝트 5: 객체 추적 시스템

### 프로젝트 개요

```
객체 추적 시스템 (Object Tracking System):
배경 차분과 칼만 필터를 조합한 다중 객체 추적

처리 흐름:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 프레임  │ → │ 배경    │ → │ 윤곽선  │ → │ 칼만    │
│ 입력    │    │ 차분    │    │ 검출    │    │ 필터    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                  │
                                                  ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 결과    │ ← │ ID      │ ← │ 헝가리안│ ← │ 예측    │
│ 출력    │    │ 할당    │    │ 매칭    │    │ 위치    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 단계별 구현

```python
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanTracker:
    """칼만 필터 기반 단일 객체 추적기"""

    def __init__(self, initial_pos):
        # 칼만 필터 초기화
        # 상태 벡터: [x, y, vx, vy]
        self.kalman = cv2.KalmanFilter(4, 2)

        # 전이 행렬 (등속 운동 모델)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 측정 행렬
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 프로세스 노이즈
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # 측정 노이즈
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

        # 초기 상태
        self.kalman.statePre = np.array([
            [initial_pos[0]],
            [initial_pos[1]],
            [0],
            [0]
        ], dtype=np.float32)

        self.kalman.statePost = self.kalman.statePre.copy()

        self.age = 0  # 추적 프레임 수
        self.hits = 1  # 성공적인 매칭 수
        self.time_since_update = 0  # 업데이트 이후 프레임 수

    def predict(self):
        """다음 위치 예측"""
        prediction = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return prediction[:2].flatten()

    def update(self, measurement):
        """측정값으로 상태 업데이트"""
        self.kalman.correct(np.array(measurement, dtype=np.float32))
        self.hits += 1
        self.time_since_update = 0

    def get_state(self):
        """현재 상태 반환"""
        return self.kalman.statePost[:2].flatten()


class MultiObjectTracker:
    """다중 객체 추적 시스템"""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.trackers = []
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # 배경 차분기
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def detect_objects(self, frame):
        """배경 차분으로 객체 검출"""
        # 배경 차분
        fg_mask = self.bg_subtractor.apply(frame)

        # 그림자 제거
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 윤곽선 검출
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 최소 면적
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                detections.append({
                    'bbox': (x, y, w, h),
                    'center': center
                })

        return detections, fg_mask

    def iou(self, bbox1, bbox2):
        """IoU (Intersection over Union) 계산"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def associate_detections(self, detections):
        """검출과 트래커 매칭 (헝가리안 알고리즘)"""
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.trackers)))

        # 비용 행렬 계산 (거리 기반)
        cost_matrix = np.zeros((len(detections), len(self.trackers)))

        for d, det in enumerate(detections):
            for t, tracker in enumerate(self.trackers):
                pred = tracker['kalman'].predict()
                dist = np.linalg.norm(np.array(det['center']) - pred)
                cost_matrix[d, t] = dist

        # 헝가리안 알고리즘으로 최적 매칭
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 100:  # 거리 임계값
                matched.append((row, col))
                unmatched_detections.remove(row)
                unmatched_trackers.remove(col)

        return matched, unmatched_detections, unmatched_trackers

    def update(self, frame):
        """추적 업데이트"""
        # 객체 검출
        detections, fg_mask = self.detect_objects(frame)

        # 예측
        for tracker in self.trackers:
            tracker['kalman'].predict()

        # 매칭
        matched, unmatched_dets, unmatched_trks = \
            self.associate_detections(detections)

        # 매칭된 트래커 업데이트
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx]['kalman'].update(
                np.array(detections[det_idx]['center'])
            )
            self.trackers[trk_idx]['bbox'] = detections[det_idx]['bbox']

        # 새 트래커 생성
        for det_idx in unmatched_dets:
            tracker = {
                'id': self.next_id,
                'kalman': KalmanTracker(detections[det_idx]['center']),
                'bbox': detections[det_idx]['bbox'],
                'color': (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
            }
            self.trackers.append(tracker)
            self.next_id += 1

        # 오래된 트래커 제거
        self.trackers = [t for t in self.trackers
                        if t['kalman'].time_since_update < self.max_age]

        # 결과 반환
        results = []
        for tracker in self.trackers:
            if tracker['kalman'].hits >= self.min_hits:
                results.append({
                    'id': tracker['id'],
                    'bbox': tracker['bbox'],
                    'color': tracker['color'],
                    'center': tracker['kalman'].get_state()
                })

        return results, fg_mask

    def draw(self, frame, results):
        """결과 시각화"""
        for obj in results:
            x, y, w, h = obj['bbox']
            color = obj['color']

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID: {obj['id']}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 궤적 표시 (중심점)
            center = tuple(obj['center'].astype(int))
            cv2.circle(frame, center, 4, color, -1)

        return frame

# 사용 예
def multi_object_tracking(video_path):
    """다중 객체 추적 실행"""

    tracker = MultiObjectTracker()
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 추적 업데이트
        results, fg_mask = tracker.update(frame)

        # 시각화
        output = tracker.draw(frame, results)

        # 정보 표시
        cv2.putText(output, f"Objects: {len(results)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Multi-Object Tracking', output)
        cv2.imshow('Foreground Mask', fg_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
# multi_object_tracking('traffic.mp4')
```

---

## 연습 문제 및 확장 아이디어

### 프로젝트 1: 문서 스캐너 확장

1. **OCR 통합**: Tesseract OCR을 연동하여 텍스트 추출
2. **자동 색상 보정**: 히스토그램 평활화로 문서 가독성 향상
3. **다중 페이지 지원**: 연속 촬영으로 PDF 생성
4. **손글씨 인식**: 손으로 쓴 문서의 디지털화
5. **영수증 파싱**: 금액, 날짜 등 자동 추출

### 프로젝트 2: 차선 검출 확장

1. **곡선 차선 검출**: 2차/3차 다항식 피팅
2. **차선 이탈 경고**: 차량 중심과 차선 중심 비교
3. **야간 모드**: 조명 조건에 따른 파라미터 조정
4. **다중 차선 검출**: 인접 차선까지 검출
5. **차량 검출 통합**: YOLO와 결합하여 앞차 감지

### 프로젝트 3: AR 마커 확장

1. **3D 모델 렌더링**: OpenGL과 연동하여 3D 객체 표시
2. **다중 마커 인터랙션**: 마커 간 관계 인식
3. **마커 없는 AR**: 평면 검출 기반 AR
4. **게임 개발**: 마커 기반 간단한 AR 게임
5. **가구 배치 시뮬레이션**: 실제 공간에 가상 가구 배치

### 프로젝트 4: 얼굴 필터 확장

1. **표정 인식**: 웃음, 눈 깜빡임 감지하여 필터 변경
2. **3D 필터**: 얼굴 포즈에 맞춰 3D 변형
3. **배경 교체**: 세그멘테이션으로 배경만 교체
4. **얼굴 스왑**: 두 사람의 얼굴 교환
5. **에이징 필터**: 얼굴 노화/젊어지기 효과

### 프로젝트 5: 객체 추적 확장

1. **Re-ID 기능**: 화면 밖으로 나갔다 들어온 객체 재식별
2. **속도 측정**: 캘리브레이션 후 실제 속도 계산
3. **영역 침입 감지**: 특정 영역 진입 시 알림
4. **궤적 분석**: 이동 패턴 분석 및 이상 감지
5. **딥러닝 통합**: YOLO + DeepSORT로 정확도 향상

---

## 다음 단계

OpenCV와 컴퓨터 비전의 기초를 마스터했습니다. 더 깊은 학습을 위해 다음 주제들을 추천합니다:

### 딥러닝 프레임워크
- **PyTorch**: 연구 및 프로토타이핑에 강력
- **TensorFlow/Keras**: 프로덕션 배포에 적합
- **ONNX**: 모델 호환성을 위한 표준

### 고급 컴퓨터 비전
- **이미지 세그멘테이션**: U-Net, Mask R-CNN
- **포즈 추정**: OpenPose, MediaPipe
- **GAN 기반 이미지 생성**: StyleGAN, Pix2Pix
- **3D 비전**: 스테레오 비전, 깊이 추정

### 응용 분야
- **자율주행**: SLAM, 센서 퓨전
- **의료 영상**: CT/MRI 분석, 질병 검출
- **산업 검사**: 결함 검출, 품질 관리
- **보안/감시**: 이상 행동 감지, 얼굴 인식

---

## 참고 자료

- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [PyImageSearch](https://pyimagesearch.com/) - 실전 프로젝트 튜토리얼
- [Learn OpenCV](https://learnopencv.com/) - 고급 예제
- [Mediapipe](https://google.github.io/mediapipe/) - Google의 ML 솔루션
- [Papers With Code](https://paperswithcode.com/) - 최신 연구 및 코드
- Bradski, G., & Kaehler, A. (2008). "Learning OpenCV"
- Szeliski, R. (2010). "Computer Vision: Algorithms and Applications"
