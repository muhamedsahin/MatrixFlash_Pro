; AVX2 4x16 microkernel. Accumulators stay in YMM0-YMM7.
; Win64: rcx=kc, rdx=A, r8=B, r9=C, [rsp+40]=ldc, [rsp+48]=beta at entry.
; A is K-major, 4 floats per k. B is K-major, 16 floats per k.

.code

nexus_avx2_kernel_4x16 PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 80
    .allocstack 80
    .endprolog

    vmovaps xmmword ptr [rsp], xmm6
    vmovaps xmmword ptr [rsp+16], xmm7
    vmovaps xmmword ptr [rsp+32], xmm8
    vmovaps xmmword ptr [rsp+48], xmm9
    vmovaps xmmword ptr [rsp+64], xmm10

    movsxd r10, dword ptr [rsp+128]
    vmovss xmm10, dword ptr [rsp+136]
    mov r11d, ecx

    vxorps xmm0, xmm0, xmm0
    vucomiss xmm10, xmm0
    jne beta_scale
    jp beta_scale

    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    jmp k_loop

beta_scale:
    vbroadcastss ymm10, xmm10
    vmulps ymm0, ymm10, ymmword ptr [r9]
    vmulps ymm1, ymm10, ymmword ptr [r9+32]
    lea rax, [r10*4]
    vmulps ymm2, ymm10, ymmword ptr [r9+rax]
    vmulps ymm3, ymm10, ymmword ptr [r9+rax+32]
    lea rbx, [r10*8]
    vmulps ymm4, ymm10, ymmword ptr [r9+rbx]
    vmulps ymm5, ymm10, ymmword ptr [r9+rbx+32]
    lea rbx, [rax+rax*2]
    vmulps ymm6, ymm10, ymmword ptr [r9+rbx]
    vmulps ymm7, ymm10, ymmword ptr [r9+rbx+32]

k_loop:
    cmp r11d, 8
    jl k_by4

k_unroll8:
    vmovups ymm8, ymmword ptr [r8]
    vmovups ymm9, ymmword ptr [r8+32]
    vbroadcastss ymm10, dword ptr [rdx]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+4]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+8]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+12]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+64]
    vmovups ymm9, ymmword ptr [r8+96]
    vbroadcastss ymm10, dword ptr [rdx+16]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+20]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+24]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+28]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+128]
    vmovups ymm9, ymmword ptr [r8+160]
    vbroadcastss ymm10, dword ptr [rdx+32]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+36]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+40]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+44]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+192]
    vmovups ymm9, ymmword ptr [r8+224]
    vbroadcastss ymm10, dword ptr [rdx+48]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+52]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+56]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+60]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+256]
    vmovups ymm9, ymmword ptr [r8+288]
    vbroadcastss ymm10, dword ptr [rdx+64]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+68]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+72]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+76]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+320]
    vmovups ymm9, ymmword ptr [r8+352]
    vbroadcastss ymm10, dword ptr [rdx+80]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+84]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+88]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+92]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+384]
    vmovups ymm9, ymmword ptr [r8+416]
    vbroadcastss ymm10, dword ptr [rdx+96]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+100]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+104]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+108]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+448]
    vmovups ymm9, ymmword ptr [r8+480]
    vbroadcastss ymm10, dword ptr [rdx+112]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+116]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+120]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+124]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    add rdx, 128
    add r8, 512
    sub r11d, 8
    cmp r11d, 8
    jge k_unroll8

k_by4:
    cmp r11d, 4
    jl k_tail

k_unroll:
    vmovups ymm8, ymmword ptr [r8]
    vmovups ymm9, ymmword ptr [r8+32]
    vbroadcastss ymm10, dword ptr [rdx]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+4]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+8]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+12]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+64]
    vmovups ymm9, ymmword ptr [r8+96]
    vbroadcastss ymm10, dword ptr [rdx+16]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+20]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+24]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+28]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+128]
    vmovups ymm9, ymmword ptr [r8+160]
    vbroadcastss ymm10, dword ptr [rdx+32]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+36]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+40]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+44]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    vmovups ymm8, ymmword ptr [r8+192]
    vmovups ymm9, ymmword ptr [r8+224]
    vbroadcastss ymm10, dword ptr [rdx+48]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+52]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+56]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+60]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    add rdx, 64
    add r8, 256
    sub r11d, 4
    cmp r11d, 4
    jge k_unroll

k_tail:
    cmp r11d, 0
    je k_store

k_one:
    vmovups ymm8, ymmword ptr [r8]
    vmovups ymm9, ymmword ptr [r8+32]
    vbroadcastss ymm10, dword ptr [rdx]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm1, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+4]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm3, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+8]
    vfmadd231ps ymm4, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9
    vbroadcastss ymm10, dword ptr [rdx+12]
    vfmadd231ps ymm6, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9
    add rdx, 16
    add r8, 64
    dec r11d
    jnz k_one

k_store:
    vmovups ymmword ptr [r9], ymm0
    vmovups ymmword ptr [r9+32], ymm1
    lea rax, [r10*4]
    vmovups ymmword ptr [r9+rax], ymm2
    vmovups ymmword ptr [r9+rax+32], ymm3
    lea rbx, [r10*8]
    vmovups ymmword ptr [r9+rbx], ymm4
    vmovups ymmword ptr [r9+rbx+32], ymm5
    lea rbx, [rax+rax*2]
    vmovups ymmword ptr [r9+rbx], ymm6
    vmovups ymmword ptr [r9+rbx+32], ymm7

    vmovaps xmm6, xmmword ptr [rsp]
    vmovaps xmm7, xmmword ptr [rsp+16]
    vmovaps xmm8, xmmword ptr [rsp+32]
    vmovaps xmm9, xmmword ptr [rsp+48]
    vmovaps xmm10, xmmword ptr [rsp+64]
    add rsp, 80
    pop rbx
    ret
nexus_avx2_kernel_4x16 ENDP

END
