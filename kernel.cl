/*
 * inputs dim = (N, C, H, W)
 * outputs dim = (C, H, W, N)
 * global_work_size = (ceil(N * C * H * W, 256))
 * local_work_size = (256)
 */
__kernel void NCHW2CHWN(
    __global float *inputs,
    __global float *outputs,
    int N, int C, int H, int W
    ) {
    int gid0 = get_global_id(0);
    int n = gid0 / (C * H * W);
    if (n < N) {
        int chw = gid0 - n * (C * H * W);
        outputs[chw * N + n] = inputs[gid0];
    }
}

/*
 * inputs dim = (C, H, W, N)
 * outputs dim = (N, C, H, W)
 * global_work_size = (ceil(N * C * H * W, 256))
 * local_work_size = (256)
 */
__kernel void CHWN2NCHW(
    __global float *inputs,
    __global float *outputs,
    int N, int C, int H, int W
    ) {
    int gid0 = get_global_id(0);
    int n = gid0 / (C * H * W);
    if (n < N) {
        int chw = gid0 - n * (C * H * W);
        outputs[gid0] = inputs[chw * N + n];
    }
}

/*
 * inputs  dim = (C, H, W, N)
 * outputs dim = (K, P, Q, N)
 * filters dim = (C, 3, 3, K)
 * bias    dim = (K)
 * global_work_size = (TP * TQ * BN * BK * 256, TK / BK, TN / BN)
 * local_work_size = (256)
 */
__kernel void winograd_2x2_3x3_16x16(
    __global float *inputs,
    __global float *outputs,
    __global float *filters,
    __global float *bias,
    int N,
    int C, int H, int W,
    int K, int P, int Q,
    int pad,
    int TP, int TQ, int BN, int BK,
    int TPmask, int TPwidth, int TPshift,
    int TQmask, int TQwidth, int TQshift,
    int Nmask, int Nwidth
    ) {
    int tptqbnbk = get_group_id(0);
    int tp = tptqbnbk / (TQ * BN * BK);
    int tqbnbk = tptqbnbk - tp * (TQ * BN * BK);
    int tq = tqbnbk / (BN * BK);
    int bnbk = tqbnbk - tq * (BN * BK);
    int bn = bnbk / (BK);
    int bk = bnbk - bn * (BK);

    int tid = get_local_id(0);
    int tidlow = tid & 15;
    int c = (tid & 0x70) >> 4;
    int ci = c - (C & 7 ? 8 - (C & 7) : 0);
    tp = (tp << TPwidth) + ((tid & TPmask) >> TPshift);
    tq = (tq << TQwidth) + ((tid & TQmask) >> TQshift);
    int h = (tp << 1) - pad, w = (tq << 1) - pad;
    int n = ((get_group_id(2) * BN + bn) << Nwidth) + (tid & Nmask);
    int k = ((get_group_id(1) * BK + bk) << 4) + tidlow;

    __local float SM[2 * 8 * 16 * 16];
    __local float *pRSV = SM + (tid & 0xf0) + (tid & 0x3);
    __local float *pRSU = SM + 8 * 16 * 16 + (tid & 0xf0) + ((tid & 0xc) >> 2);

    float r[4][4], rA[4], rB[4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            r[i][j] = 0;
        }
    }

    if (tid < 128) { // image transform
        float v[4][4], TV[4][4], V[4][4];

        bool preds[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                preds[i][j] = n < N && 0 <= h + i && h + i < H && 0 <= w + j && w + j < W;
            }
        }

        __global float *pV = inputs + ((ci * H + h) * W + w) * N + n;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v[i][j] = ci >= 0 && preds[i][j] ? pV[(i * W + j) * N] : 0;
            }
        }

        __local float *pWSV = SM + c * 16 * 16 + tidlow;
        while (true) {
            TV[0][0] = v[0][0] - v[2][0];
            TV[0][1] = v[0][1] - v[2][1];
            TV[0][2] = v[0][2] - v[2][2];
            TV[0][3] = v[0][3] - v[2][3];

            TV[3][0] = v[1][0] - v[3][0];
            TV[3][1] = v[1][1] - v[3][1];
            TV[3][2] = v[1][2] - v[3][2];
            TV[3][3] = v[1][3] - v[3][3];

            TV[1][0] = v[1][0] + v[2][0];
            TV[1][1] = v[1][1] + v[2][1];
            TV[1][2] = v[1][2] + v[2][2];
            TV[1][3] = v[1][3] + v[2][3];

            TV[2][0] = v[2][0] - v[1][0];
            TV[2][1] = v[2][1] - v[1][1];
            TV[2][2] = v[2][2] - v[1][2];
            TV[2][3] = v[2][3] - v[1][3];

            V[0][0] = TV[0][0] - TV[0][2];
            V[0][3] = TV[0][1] - TV[0][3];
            V[3][0] = TV[3][0] - TV[3][2];
            V[3][3] = TV[3][1] - TV[3][3];

            V[1][0] = TV[1][0] - TV[1][2];
            V[2][0] = TV[2][0] - TV[2][2];
            V[1][3] = TV[1][1] - TV[1][3];
            V[2][3] = TV[2][1] - TV[2][3];

            V[2][1] = TV[2][1] + TV[2][2];
            V[2][2] = TV[2][2] - TV[2][1];

            V[0][1] = TV[0][1] + TV[0][2];
            V[0][2] = TV[0][2] - TV[0][1];
            V[1][1] = TV[1][1] + TV[1][2];
            V[1][2] = TV[1][2] - TV[1][1];
            V[3][1] = TV[3][1] + TV[3][2];
            V[3][2] = TV[3][2] - TV[3][1];

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    pWSV[(i * 4 + j) * 16] = V[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int l = 0; l < 8; ++l) {
                for (int i = 0; i < 4; ++i) {
                    rA[i] = pRSU[l * 16 * 16 + i * 4];
                    rB[i] = pRSV[l * 16 * 16 + i * 4];
                }
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        r[i][j] += rA[i] * rB[j];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            ci += 8;
            if (ci >= C) break;
            pV += 8 * H * W * N;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    v[i][j] = preds[i][j] ? pV[(i * W + j) * N] : 0;
                }
            }
        }
    } else { // filter transform
        float u[3][3], TU[4][3], TA[3], TB[4], U[4][4];

        bool pred = k < K;

        __global float *pU = filters + ci * 3 * 3 * K + k;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                u[i][j] = ci >= 0 && pred ? pU[(i * 3 + j) * K] : 0;
            }
        }

        __local float *pWSU = SM + (c + 8) * 16 * 16 + tidlow;
        while (true) {
            TA[0] = (u[0][0] + u[2][0]) * 0.5;
            TA[1] = (u[0][1] + u[2][1]) * 0.5;
            TA[2] = (u[0][2] + u[2][2]) * 0.5;
            TU[0][0] = u[0][0];
            TU[0][1] = u[0][1];
            TU[0][2] = u[0][2];
            TU[3][0] = u[2][0];
            TU[3][1] = u[2][1];
            TU[3][2] = u[2][2];
            TU[1][0] = TA[0] + u[1][0] * 0.5;
            TU[2][0] = TA[0] - u[1][0] * 0.5;
            TU[1][1] = TA[1] + u[1][1] * 0.5;
            TU[2][1] = TA[1] - u[1][1] * 0.5;
            TU[1][2] = TA[2] + u[1][2] * 0.5;
            TU[2][2] = TA[2] - u[1][2] * 0.5;
            TB[0] = (TU[0][0] + TU[0][2]) * 0.5;
            TB[1] = (TU[1][0] + TU[1][2]) * 0.5;
            TB[2] = (TU[2][0] + TU[2][2]) * 0.5;
            TB[3] = (TU[3][0] + TU[3][2]) * 0.5;
            U[0][0] = TU[0][0];
            U[0][3] = TU[0][2];
            U[3][0] = TU[3][0];
            U[3][3] = TU[3][2];
            U[1][0] = TU[1][0];
            U[2][0] = TU[2][0];
            U[1][3] = TU[1][2];
            U[2][3] = TU[2][2];
            U[1][1] = TB[1] + TU[1][1] * 0.5;
            U[1][2] = TB[1] - TU[1][1] * 0.5;
            U[2][1] = TB[2] + TU[2][1] * 0.5;
            U[2][2] = TB[2] - TU[2][1] * 0.5;
            U[0][1] = TB[0] + TU[0][1] * 0.5;
            U[0][2] = TB[0] - TU[0][1] * 0.5;
            U[3][1] = TB[3] + TU[3][1] * 0.5;
            U[3][2] = TB[3] - TU[3][1] * 0.5;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    pWSU[(i * 4 + j) * 16] = U[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int l = 0; l < 8; ++l) {
                for (int i = 0; i < 4; ++i) {
                    rA[i] = pRSU[l * 16 * 16 + i * 4];
                    rB[i] = pRSV[l * 16 * 16 + i * 4];
                }
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        r[i][j] += rA[i] * rB[j];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            ci += 8;
            if (ci >= C) break;
            pU += 8 * 3 * 3 * K;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    u[i][j] = pred ? pU[(i * 3 + j) * K] : 0;
                }
            }
        }
    }

    // inverse transform
    {
        // log(16 * 16) - 2, log(16) - 4
        __local float *pWSM = SM + ((tid & 0x0c) << 6) + ((tid & 0xf0) << 0) + (tid & 0x03);
        __local float *pRSM = SM + ((tid & 0xf0) << 4) + tidlow;
        int oh = h + pad, ow = w + pad, on = n;
        int ok = k - tidlow + ((tid & 0xf0) >> 4);
        __global float *pO = outputs + ((ok * P + oh) * Q + ow) * N + on;

        bool preds[2][2];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                preds[i][j] = on < N && 0 <= oh + i && oh + i < P && 0 <= ow + j && ow + j < Q;
            }
        }

        {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    // log(4 * 16 * 16)
                    pWSM[(i << 10) + (j << 2)] = r[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            float m[4][4], TM[4][2], M[2][2];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    m[i][j] = pRSM[(i * 4 + j) * 16];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            TM[0][0] = m[0][0] + m[0][1] + m[0][2];
            TM[0][1] = m[0][1] - m[0][2] - m[0][3];
            TM[1][0] = m[1][0] + m[1][1] + m[1][2];
            TM[1][1] = m[1][1] - m[1][2] - m[1][3];
            TM[2][0] = m[2][0] + m[2][1] + m[2][2];
            TM[2][1] = m[2][1] - m[2][2] - m[2][3];
            TM[3][0] = m[3][0] + m[3][1] + m[3][2];
            TM[3][1] = m[3][1] - m[3][2] - m[3][3];

            M[0][0] = TM[0][0] + TM[1][0] + TM[2][0];
            M[0][1] = TM[0][1] + TM[1][1] + TM[2][1];
            M[1][0] = TM[1][0] - TM[2][0] - TM[3][0];
            M[1][1] = TM[1][1] - TM[2][1] - TM[3][1];

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    if (ok < K && preds[i][j]) {
                        pO[(i * Q + j) * N] = M[i][j] + bias[ok];
                    }
                }
            }
        }
    }
}

/*
 * inputs  dim = (C, H, W, N)
 * outputs dim = (K, P, Q, N)
 * filters dim = (C, 3, 3, K)
 * bias    dim = (K)
 * global_work_size = (TP * TQ * BN * BK * 256, TK / BK, TN / BN)
 * local_work_size = (256)
 */
__kernel void winograd_2x2_3x3_32x32(
    __global float *inputs,
    __global float *outputs,
    __global float *filters,
    __global float *bias,
    int N,
    int C, int H, int W,
    int K, int P, int Q,
    int pad,
    int TP, int TQ, int BN, int BK,
    int TPmask, int TPwidth, int TPshift,
    int TQmask, int TQwidth, int TQshift,
    int Nmask, int Nwidth
    ) {
    int tptqbnbk = get_group_id(0);
    int tp = tptqbnbk / (TQ * BN * BK);
    int tqbnbk = tptqbnbk - tp * (TQ * BN * BK);
    int tq = tqbnbk / (BN * BK);
    int bnbk = tqbnbk - tq * (BN * BK);
    int bn = bnbk / (BK);
    int bk = bnbk - bn * (BK);

    int tid = get_local_id(0);
    int tid32 = tid & 31;
    int c = (tid & 0x60) >> 5; // 01100000
    int ci = c - (C & 3 ? 4 - (C & 3) : 0);
    tp = (tp << TPwidth) + ((tid & TPmask) >> TPshift);
    tq = (tq << TQwidth) + ((tid & TQmask) >> TQshift);
    int h = (tp << 1) - pad, w = (tq << 1) - pad;
    int n = ((get_group_id(2) * BN + bn) << Nwidth) + (tid & Nmask);
    int k = ((get_group_id(1) * BK + bk) << 5) + tid32;

    __local float SM[8 * 16 * 32];
    __local float *pRSV = SM + ((tid & 0xf0) << 1) + (tid & 0x3);
    __local float *pRSU = SM + 4 * 16 * 32 + ((tid & 0xf0) << 1) + ((tid & 0xc) >> 2);

    float r[8][8], rA[8], rB[8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            r[i][j] = 0;
        }
    }

    if (tid < 128) { // image transform
        float v[4][4], TV[4][4], V[4][4];

        bool preds[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                preds[i][j] = n < N && 0 <= h + i && h + i < H && 0 <= w + j && w + j < W;
            }
        }

        __global float *pV = inputs + ((ci * H + h) * W + w) * N + n;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v[i][j] = ci >= 0 && preds[i][j] ? pV[(i * W + j) * N] : 0;
            }
        }

        __local float *pWSV = SM + c * 16 * 32 + tid32;
        while (true) {
            TV[0][0] = v[0][0] - v[2][0];
            TV[0][1] = v[0][1] - v[2][1];
            TV[0][2] = v[0][2] - v[2][2];
            TV[0][3] = v[0][3] - v[2][3];

            TV[3][0] = v[1][0] - v[3][0];
            TV[3][1] = v[1][1] - v[3][1];
            TV[3][2] = v[1][2] - v[3][2];
            TV[3][3] = v[1][3] - v[3][3];

            TV[1][0] = v[1][0] + v[2][0];
            TV[1][1] = v[1][1] + v[2][1];
            TV[1][2] = v[1][2] + v[2][2];
            TV[1][3] = v[1][3] + v[2][3];

            TV[2][0] = v[2][0] - v[1][0];
            TV[2][1] = v[2][1] - v[1][1];
            TV[2][2] = v[2][2] - v[1][2];
            TV[2][3] = v[2][3] - v[1][3];

            V[0][0] = TV[0][0] - TV[0][2];
            V[0][3] = TV[0][1] - TV[0][3];
            V[3][0] = TV[3][0] - TV[3][2];
            V[3][3] = TV[3][1] - TV[3][3];

            V[1][0] = TV[1][0] - TV[1][2];
            V[2][0] = TV[2][0] - TV[2][2];
            V[1][3] = TV[1][1] - TV[1][3];
            V[2][3] = TV[2][1] - TV[2][3];

            V[2][1] = TV[2][1] + TV[2][2];
            V[2][2] = TV[2][2] - TV[2][1];

            V[0][1] = TV[0][1] + TV[0][2];
            V[0][2] = TV[0][2] - TV[0][1];
            V[1][1] = TV[1][1] + TV[1][2];
            V[1][2] = TV[1][2] - TV[1][1];
            V[3][1] = TV[3][1] + TV[3][2];
            V[3][2] = TV[3][2] - TV[3][1];

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    pWSV[(i * 4 + j) * 32] = V[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int l = 0; l < 4; ++l) {
                for (int i = 0; i < 8; ++i) {
                    rA[i] = pRSU[l * 512 + i * 4];
                    rB[i] = pRSV[l * 512 + i * 4];
                }
                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        r[i][j] += rA[i] * rB[j];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            ci += 4;
            if (ci >= C) break;
            pV += 4 * H * W * N;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    v[i][j] = preds[i][j] ? pV[(i * W + j) * N] : 0;
                }
            }
        }
    } else { // filter transform
        float u[3][3], TU[4][3], TA[3], TB[4], U[4][4];

        bool pred = k < K;

        __global float *pU = filters + ci * 3 * 3 * K + k;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                u[i][j] = ci >= 0 && pred ? pU[(i * 3 + j) * K] : 0;
            }
        }

        __local float *pWSU = SM + (c + 4) * 16 * 32 + tid32;
        while (true) {
            TA[0] = (u[0][0] + u[2][0]) * 0.5;
            TA[1] = (u[0][1] + u[2][1]) * 0.5;
            TA[2] = (u[0][2] + u[2][2]) * 0.5;
            TU[0][0] = u[0][0];
            TU[0][1] = u[0][1];
            TU[0][2] = u[0][2];
            TU[3][0] = u[2][0];
            TU[3][1] = u[2][1];
            TU[3][2] = u[2][2];
            TU[1][0] = TA[0] + u[1][0] * 0.5;
            TU[2][0] = TA[0] - u[1][0] * 0.5;
            TU[1][1] = TA[1] + u[1][1] * 0.5;
            TU[2][1] = TA[1] - u[1][1] * 0.5;
            TU[1][2] = TA[2] + u[1][2] * 0.5;
            TU[2][2] = TA[2] - u[1][2] * 0.5;
            TB[0] = (TU[0][0] + TU[0][2]) * 0.5;
            TB[1] = (TU[1][0] + TU[1][2]) * 0.5;
            TB[2] = (TU[2][0] + TU[2][2]) * 0.5;
            TB[3] = (TU[3][0] + TU[3][2]) * 0.5;
            U[0][0] = TU[0][0];
            U[0][3] = TU[0][2];
            U[3][0] = TU[3][0];
            U[3][3] = TU[3][2];
            U[1][0] = TU[1][0];
            U[2][0] = TU[2][0];
            U[1][3] = TU[1][2];
            U[2][3] = TU[2][2];
            U[1][1] = TB[1] + TU[1][1] * 0.5;
            U[1][2] = TB[1] - TU[1][1] * 0.5;
            U[2][1] = TB[2] + TU[2][1] * 0.5;
            U[2][2] = TB[2] - TU[2][1] * 0.5;
            U[0][1] = TB[0] + TU[0][1] * 0.5;
            U[0][2] = TB[0] - TU[0][1] * 0.5;
            U[3][1] = TB[3] + TU[3][1] * 0.5;
            U[3][2] = TB[3] - TU[3][1] * 0.5;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    pWSU[(i * 4 + j) * 32] = U[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int l = 0; l < 4; ++l) {
                for (int i = 0; i < 8; ++i) {
                    rA[i] = pRSU[l * 512 + i * 4];
                    rB[i] = pRSV[l * 512 + i * 4];
                }
                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        r[i][j] += rA[i] * rB[j];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            ci += 4;
            if (ci >= C) break;
            pU += 4 * 3 * 3 * K;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    u[i][j] = pred ? pU[(i * 3 + j) * K] : 0;
                }
            }
        }
    }

    // inverse transform
    {
        __local float *pWSM = SM + ((tid & 0x0c) << 7) + ((tid & 0xf0) << 1) + (tid & 0x03);
        __local float *pRSM = SM + ((tid & 0xe0) << 4) + tid32;
        int oh = h + pad, ow = w + pad, on = n;
        int ok = k - tid32 + ((tid & 0xe0) >> 5);
        __global float *pO = outputs + ((ok * P + oh) * Q + ow) * N + on;

        bool preds[2][2];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                preds[i][j] = on < N && 0 <= oh + i && oh + i < P && 0 <= ow + j && ow + j < Q;
            }
        }

        for (int l = 0; l < 4; ++l) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 8; ++j) {
                    pWSM[(i << 11) + (j << 2)] = r[l * 2 + i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            float m[4][4], TM[4][2], M[2][2];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    m[i][j] = pRSM[(i * 4 + j) * 32];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            TM[0][0] = m[0][0] + m[0][1] + m[0][2];
            TM[0][1] = m[0][1] - m[0][2] - m[0][3];
            TM[1][0] = m[1][0] + m[1][1] + m[1][2];
            TM[1][1] = m[1][1] - m[1][2] - m[1][3];
            TM[2][0] = m[2][0] + m[2][1] + m[2][2];
            TM[2][1] = m[2][1] - m[2][2] - m[2][3];
            TM[3][0] = m[3][0] + m[3][1] + m[3][2];
            TM[3][1] = m[3][1] - m[3][2] - m[3][3];

            M[0][0] = TM[0][0] + TM[1][0] + TM[2][0];
            M[0][1] = TM[0][1] + TM[1][1] + TM[2][1];
            M[1][0] = TM[1][0] - TM[2][0] - TM[3][0];
            M[1][1] = TM[1][1] - TM[2][1] - TM[3][1];

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    if (ok < K && preds[i][j]) {
                        pO[(i * Q + j) * N] = M[i][j] + bias[ok];
                    }
                }
            }
            ok += 8;
            pO += 8 * P * Q * N;
        }
    }
}

typedef float NNType;
__kernel void CCF(
        __global NNType * inputs,
        __global NNType * outputs,
        __global NNType * filters,
        __global NNType * bias,
        uint width,
        uint height,
        uint depth,
        uint prev_width,
        uint prev_height,
        uint prev_depth,
        uint filter_width,
        uint filter_height,
        uint padding_size,
        uint stride)
{
    uint j = get_global_id(0) % width;
    uint i = get_global_id(0) / width;
    uint a = get_global_id(1);
    uint batch_id = get_global_id(2);
    __global NNType * input = inputs + batch_id * prev_depth * prev_width * prev_height;
    __global NNType * output = outputs + batch_id * depth * width * height;
    uint from_i = i * stride;
    uint from_j = j * stride;
    //for(uint a = 0; a < depth; a++)
    {
        NNType sum = 0;
        for(uint b = 0; b < prev_depth; b++)
        {
            for(uint fi = 0; fi < filter_height; fi++)
            {
                for(uint fj = 0; fj < filter_width; fj++)
                {
                    int iin = -padding_size + from_i + fi;
                    int jin = -padding_size + from_j + fj;
                    NNType x = (iin >= 0 && iin < prev_height && jin >= 0 && jin < prev_width)
                        ? input[b * prev_width * prev_height + iin * prev_width + jin]
                        : 0;
                    NNType f = filters[ (a * prev_depth + b) * (filter_height * filter_width) + fi * filter_width + fj ];
                    sum += x * f;
                }
            }
        }
        sum += bias[a];
        output[a * width * height + i * width + j] = sum;
    }
}

#define CONV_BK 16
#define CONV_BKH 8
#define CONV_BIJ 64
#define CONV_LXY 16
#define CONV_RXY 4

// if x <= 4608 = 512 * 9
#define DIV3(x) (((x)*5462)>>14)

#define GEMM44(u, v) \
    rA[0] = smA[u][lx * CONV_RXY]; \
    rA[1] = smA[u][lx * CONV_RXY + 1]; \
    rB[0] = smB[u][ly * CONV_RXY]; \
    rB[1] = smB[u][ly * CONV_RXY + 1]; \
    for (int ki = u; ki < v; ++ki) { \
        rB[2] = smB[ki][ly * CONV_RXY + 2]; \
        rB[3] = smB[ki][ly * CONV_RXY + 3]; \
        accum[0] += rA[0] * rB[0]; \
        accum[1] += rA[0] * rB[1]; \
        accum[4] += rA[1] * rB[0]; \
        accum[5] += rA[1] * rB[1]; \
        rA[2] = smA[ki][lx * CONV_RXY + 2]; \
        rA[3] = smA[ki][lx * CONV_RXY + 3]; \
        accum[2] += rA[0] * rB[2]; \
        accum[3] += rA[0] * rB[3]; \
        accum[6] += rA[1] * rB[2]; \
        accum[7] += rA[1] * rB[3]; \
        rA[0] = smA[ki + 1][lx * CONV_RXY]; \
        rA[1] = smA[ki + 1][lx * CONV_RXY + 1]; \
        accum[8] += rA[2] * rB[0]; \
        accum[9] += rA[2] * rB[1]; \
        accum[12] += rA[3] * rB[0]; \
        accum[13] += rA[3] * rB[1]; \
        rB[0] = smB[ki + 1][ly * CONV_RXY]; \
        rB[1] = smB[ki + 1][ly * CONV_RXY + 1]; \
        accum[10] += rA[2] * rB[2]; \
        accum[11] += rA[2] * rB[3]; \
        accum[14] += rA[3] * rB[2]; \
        accum[15] += rA[3] * rB[3]; \
    }

// global size IA * KA
// local size 256
__kernel void conv_preA(__global float *AU, __global float *A, int KU, int IA, int KA) {
    int gid = get_global_id(0);
    int bn = gid / (KA * CONV_BIJ), bo = gid - bn * (KA * CONV_BIJ);
    int k = bo / CONV_BIJ, i = bo % CONV_BIJ + bn * CONV_BIJ;
    A[gid] = k < KU ? AU[i * KU + k] : 0.0f;
}

// global size JA, IA, batch
// local size 256, 1, 1
__kernel void conv_postC(__global float *C, __global float *CU, int IA, int JA, int IU, int JU) {
    int c = get_global_id(2), i = get_global_id(1), j = get_global_id(0);
    if (j < JU) {
        CU[(c * IU + i) * JU + j] = C[(c * IA + i) * JA + j];
    }
}

// global size (JA / CONV_RXY), (IA / CONV_RXY), batch
// local size 16, 16, 1
__kernel void conv(__global float *A, __global float *B, __global float *C, __global float *D, int K, int IA, int JA, int KA, int CH, int N) {
    // +1 prevent overflow in innermost loop
    __local float smA[CONV_BK + 1][CONV_BIJ];
    __local float smB[CONV_BK + 1][CONV_BIJ];
    float rA[CONV_RXY], rB[CONV_RXY], accum[CONV_RXY * CONV_RXY] = {0};
    int gb = get_group_id(2);
    int gi = get_group_id(1), gj = get_group_id(0);
    int lx = get_local_id(1), ly = get_local_id(0);
    int lid = lx * CONV_LXY + ly;
    // CONV_BIJ
    int smx = lid >> 6, smy = lid & 63;
    int jj = gj * CONV_BIJ + smy, jf = jj < N * N;
    int hb = jj / N - 1, wb = jj % N - 1;
    int kk = smx, kk3, kk9, h, w;
    A += (gi * KA + smx) * CONV_BIJ + smy;
    B += gb * CH * N * N;

    smA[smx][smy] = A[0];
    kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
    smB[smx][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
    smA[smx + 4][smy] = A[CONV_BIJ * 4];
    kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
    smB[smx + 4][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
    for (; ; --K) {
        barrier(CLK_LOCAL_MEM_FENCE);

        smA[smx + 8][smy] = A[CONV_BIJ * 8];
        kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
        smB[smx + 8][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;

        GEMM44(0, 4);

        smA[smx + 12][smy] = A[CONV_BIJ * 12];
        kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
        smB[smx + 12][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
        A += CONV_BIJ * 16;

        GEMM44(4, 8);

        barrier(CLK_LOCAL_MEM_FENCE);

        if (K > 1) {
            smA[smx][smy] = A[0];
            kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
            smB[smx][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
        }

        GEMM44(8, 12);

        if (K > 1) {
            smA[smx + 4][smy] = A[CONV_BIJ * 4];
            kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
            smB[smx + 4][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
        }

        GEMM44(12, 16);

        if (K == 1) break;
    }
    C += (gb * IA + gi * CONV_BIJ + lx * CONV_RXY) * JA + gj * CONV_BIJ + ly * CONV_RXY;
    D += gi * CONV_BIJ + lx * CONV_RXY;
    C[0] = max(accum[0] + D[0], 0.0f);
    C[1] = max(accum[1] + D[0], 0.0f);
    C[2] = max(accum[2] + D[0], 0.0f);
    C[3] = max(accum[3] + D[0], 0.0f);
    C[JA] = max(accum[4] + D[1], 0.0f);
    C[JA + 1] = max(accum[5] + D[1], 0.0f);
    C[JA + 2] = max(accum[6] + D[1], 0.0f);
    C[JA + 3] = max(accum[7] + D[1], 0.0f);
    C[JA * 2] = max(accum[8] + D[2], 0.0f);
    C[JA * 2 + 1] = max(accum[9] + D[2], 0.0f);
    C[JA * 2 + 2] = max(accum[10] + D[2], 0.0f);
    C[JA * 2 + 3] = max(accum[11] + D[2], 0.0f);
    C[JA * 3] = max(accum[12] + D[3], 0.0f);
    C[JA * 3 + 1] = max(accum[13] + D[3], 0.0f);
    C[JA * 3 + 2] = max(accum[14] + D[3], 0.0f);
    C[JA * 3 + 3] = max(accum[15] + D[3], 0.0f);
}

/*
 * inputs dim = (N, C, H, W)
 * outputs dim = (16, C, N, TP, TQ)
 * global_work_size = {_ceil(N * C * TP * TQ, 256)}
 * local_work_size = {256}
 */
__kernel void winograd_2x2_3x3_data_transform(
    __global float *inputs,
    __global float *outputs,
    int N, int C, int H, int W,
    int pad,
    int TP, int TQ
    ) {
    int nctptq = get_global_id(0);
    int n = nctptq / (C * TP * TQ);
    if (n >= N) return;
    int ctptq = nctptq - n * (C * TP * TQ);
    int c = ctptq / (TP * TQ);
    int tptq = ctptq - c * (TP * TQ);
    int tp = tptq / (TQ);
    int tq = tptq - tp * (TQ);
    int h = tp * 2 - pad, w = tq * 2 - pad;
    float v[4][4], TV[4][4], V[4][4];

    inputs += ((n * C + c) * H + h) * W + w;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            v[i][j] = 0 <= h + i && h + i < H && 0 <= w + j && w + j < W ? inputs[i * W + j] : 0;
        }
    }

    TV[0][0] = v[0][0] - v[2][0];
    TV[0][1] = v[0][1] - v[2][1];
    TV[0][2] = v[0][2] - v[2][2];
    TV[0][3] = v[0][3] - v[2][3];
    TV[1][0] = v[1][0] + v[2][0];
    TV[1][1] = v[1][1] + v[2][1];
    TV[1][2] = v[1][2] + v[2][2];
    TV[1][3] = v[1][3] + v[2][3];
    TV[2][0] = v[2][0] - v[1][0];
    TV[2][1] = v[2][1] - v[1][1];
    TV[2][2] = v[2][2] - v[1][2];
    TV[2][3] = v[2][3] - v[1][3];
    TV[3][0] = v[1][0] - v[3][0];
    TV[3][1] = v[1][1] - v[3][1];
    TV[3][2] = v[1][2] - v[3][2];
    TV[3][3] = v[1][3] - v[3][3];

    V[0][0] = TV[0][0] - TV[0][2];
    V[0][1] = TV[0][1] + TV[0][2];
    V[0][2] = TV[0][2] - TV[0][1];
    V[0][3] = TV[0][1] - TV[0][3];
    V[1][0] = TV[1][0] - TV[1][2];
    V[1][1] = TV[1][1] + TV[1][2];
    V[1][2] = TV[1][2] - TV[1][1];
    V[1][3] = TV[1][1] - TV[1][3];
    V[2][0] = TV[2][0] - TV[2][2];
    V[2][1] = TV[2][1] + TV[2][2];
    V[2][2] = TV[2][2] - TV[2][1];
    V[2][3] = TV[2][1] - TV[2][3];
    V[3][0] = TV[3][0] - TV[3][2];
    V[3][1] = TV[3][1] + TV[3][2];
    V[3][2] = TV[3][2] - TV[3][1];
    V[3][3] = TV[3][1] - TV[3][3];

    outputs += ((c * N + n) * TP + tp) * TQ + tq;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            outputs[0] = V[i][j];
            outputs += C * N * TP * TQ;
        }
    }
}

/*
 * inputs dim = (K, C, 3, 3)
 * outputs dim = (16, K, C)
 * global_work_size = {_ceil(K * C, 256)}
 * local_work_size = {256}
 */
__kernel void winograd_2x2_3x3_filter_transform(
    __global float *inputs,
    __global float *outputs,
    int K, int C
    ) {
    int kc = get_global_id(0);
    int k = kc / (C);
    if (k >= K) return;
    int c = kc - k * (C);
    float u[3][3], TU[4][3], TA[3], TB[4], U[4][4];

    inputs += (k * C + c) * 3 * 3;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            u[i][j] = inputs[i * 3 + j];
        }
    }

    TA[0] = (u[0][0] + u[2][0]) * 0.5;
    TA[1] = (u[0][1] + u[2][1]) * 0.5;
    TA[2] = (u[0][2] + u[2][2]) * 0.5;
    TU[0][0] = u[0][0];
    TU[0][1] = u[0][1];
    TU[0][2] = u[0][2];
    TU[3][0] = u[2][0];
    TU[3][1] = u[2][1];
    TU[3][2] = u[2][2];
    TU[1][0] = TA[0] + u[1][0] * 0.5;
    TU[2][0] = TA[0] - u[1][0] * 0.5;
    TU[1][1] = TA[1] + u[1][1] * 0.5;
    TU[2][1] = TA[1] - u[1][1] * 0.5;
    TU[1][2] = TA[2] + u[1][2] * 0.5;
    TU[2][2] = TA[2] - u[1][2] * 0.5;
    TB[0] = (TU[0][0] + TU[0][2]) * 0.5;
    TB[1] = (TU[1][0] + TU[1][2]) * 0.5;
    TB[2] = (TU[2][0] + TU[2][2]) * 0.5;
    TB[3] = (TU[3][0] + TU[3][2]) * 0.5;
    U[0][0] = TU[0][0];
    U[0][3] = TU[0][2];
    U[1][0] = TU[1][0];
    U[1][3] = TU[1][2];
    U[2][0] = TU[2][0];
    U[2][3] = TU[2][2];
    U[3][0] = TU[3][0];
    U[3][3] = TU[3][2];
    U[0][1] = TB[0] + TU[0][1] * 0.5;
    U[0][2] = TB[0] - TU[0][1] * 0.5;
    U[1][1] = TB[1] + TU[1][1] * 0.5;
    U[1][2] = TB[1] - TU[1][1] * 0.5;
    U[2][1] = TB[2] + TU[2][1] * 0.5;
    U[2][2] = TB[2] - TU[2][1] * 0.5;
    U[3][1] = TB[3] + TU[3][1] * 0.5;
    U[3][2] = TB[3] - TU[3][1] * 0.5;

    outputs += k * C + c;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            outputs[0] = U[i][j];
            outputs += K * C;
        }
    }
}

/*
 * inputs dim = (16, K, N, TP, TQ)
 * outputs dim = (N, K, P, Q)
 * global_work_size = {_ceil(K * N * TP * TQ, 256)}
 * local_work_size = {256}
 */
__kernel void winograd_2x2_3x3_inverse_transform(
    __global float *inputs,
    __global float *outputs,
    __global float *bias,
    int N, int K, int P, int Q,
    int TP, int TQ
    ) {
    int kntptq = get_global_id(0);
    int k = kntptq / (N * TP * TQ);
    if (k >= K) return;
    int ntptq = kntptq - k * (N * TP * TQ);
    int n = ntptq / (TP * TQ);
    int tptq = ntptq - n * (TP * TQ);
    int tp = tptq / (TQ);
    int tq = tptq - tp * (TQ);
    int p = tp * 2, q = tq * 2;
    float m[4][4], TM[4][2], M[2][2];

    inputs += ((k * N + n) * TP + tp) * TQ + tq;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            m[i][j] = inputs[0];
            inputs += K * N * TP * TQ;
        }
    }

    TM[0][0] = m[0][0] + m[0][1] + m[0][2];
    TM[0][1] = m[0][1] - m[0][2] - m[0][3];
    TM[1][0] = m[1][0] + m[1][1] + m[1][2];
    TM[1][1] = m[1][1] - m[1][2] - m[1][3];
    TM[2][0] = m[2][0] + m[2][1] + m[2][2];
    TM[2][1] = m[2][1] - m[2][2] - m[2][3];
    TM[3][0] = m[3][0] + m[3][1] + m[3][2];
    TM[3][1] = m[3][1] - m[3][2] - m[3][3];

    M[0][0] = TM[0][0] + TM[1][0] + TM[2][0];
    M[0][1] = TM[0][1] + TM[1][1] + TM[2][1];
    M[1][0] = TM[1][0] - TM[2][0] - TM[3][0];
    M[1][1] = TM[1][1] - TM[2][1] - TM[3][1];

    outputs += ((n * K + k) * P + p) * Q + q;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (p + i < P && q + j < Q) {
                outputs[i * Q + j] = M[i][j] + bias[k];
            }
        }
    }
}
