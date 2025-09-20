def merge_maps(grid1, meta1, grid2, meta2):
    # 1) 해상도 동일성 검사
    res1, res2 = meta1['resolution'], meta2['resolution']
    assert abs(res1 - res2) < 1e-12, "두 지도 resolution이 달라 병합할 수 없습니다."
    res = res1

    # 2) 월드 좌표 범위 계산
    h1, w1 = grid1.shape
    h2, w2 = grid2.shape
    x1_min, x1_max = meta1['origin_x'], meta1['origin_x'] + w1 * res
    y1_min, y1_max = meta1['origin_y'], meta1['origin_y'] + h1 * res
    x2_min, x2_max = meta2['origin_x'], meta2['origin_x'] + w2 * res
    y2_min, y2_max = meta2['origin_y'], meta2['origin_y'] + h2 * res

    # 전체 바운딩 박스
    x_min, x_max = min(x1_min, x2_min), max(x1_max, x2_max)
    y_min, y_max = min(y1_min, y2_min), max(y1_max, y2_max)

    W = int(np.ceil((x_max - x_min) / res))
    H = int(np.ceil((y_max - y_min) / res))

    merged = -1 * np.ones((H, W), dtype=int)  # unknown으로 초기화

    def paste_union(src_grid, src_meta):
        """덮어쓰기 대신 '합집합' 규칙으로 병합."""
        ox, oy = src_meta['origin_x'], src_meta['origin_y']
        x0 = int(np.floor((ox - x_min) / res))
        y0 = int(np.floor((oy - y_min) / res))
        h, w = src_grid.shape

        # 대상 슬라이스
        dst = merged[y0:y0+h, x0:x0+w]
        a = dst
        b = src_grid

        a_unknown = (a == -1)
        b_unknown = (b == -1)

        # 점유 판정(스키마 상이해도 임계치 50 기준으로 통일)
        #  - known & value>=50 -> occupied로 간주
        a_occ = (~a_unknown) & (a >= 50)
        b_occ = (~b_unknown) & (b >= 50)

        # 적어도 하나가 점유면 점유
        occ = a_occ | b_occ

        # 알려진 값이 하나라도 있되 점유는 아닌 곳 -> free
        known = (~a_unknown) | (~b_unknown)
        free = known & (~occ)

        out = -1 * np.ones_like(a)
        out[occ] = 100   # occupied
        out[free] = 0    # free

        dst[:, :] = out  # 덮어쓰기 아님: 규칙 결과로 결합

    paste_union(grid1, meta1)  # 먼저 1번 지도 반영
    paste_union(grid2, meta2)  # 그 위에 '합집합' 규칙으로 2번 지도 결합

    merged_meta = {'resolution': res, 'origin_x': x_min, 'origin_y': y_min}
    return merged, merged_meta
