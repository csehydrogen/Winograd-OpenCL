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

