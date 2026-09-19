'use client'

import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame } from '@react-three/fiber'
import { Float } from '@react-three/drei'
import { cn } from '@/lib/utils'

const PRIMARY = '#4ade80'
const PRIMARY_DIM = '#166534'
const ACCENT = '#67e8f9'

const LATTICE = 4 // 4x4x4 lattice of matrix cells

/** Instanced 4x4x4 lattice of glowing matrix cells + wireframe shell. */
function MatrixLattice() {
  const instRef = useRef<THREE.InstancedMesh>(null!)
  const count = LATTICE ** 3

  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = useMemo(() => new THREE.Color(), [])

  useFrame((state, delta) => {
    const inst = instRef.current
    if (!inst) return

    // Slow auto-rotation...
    inst.rotation.y += delta * 0.18
    // ...plus mouse-reactive damping (pointer comes from R3F state).
    inst.rotation.x = THREE.MathUtils.damp(
      inst.rotation.x,
      -state.pointer.y * 0.55,
      2.5,
      delta,
    )
    inst.rotation.z = THREE.MathUtils.damp(
      inst.rotation.z,
      state.pointer.x * 0.18,
      2.5,
      delta,
    )
    // Subtle scale "breathing" driven by pointer speed.
    const targetScale = 1 + Math.abs(state.pointer.x) * 0.04
    inst.scale.setScalar(THREE.MathUtils.damp(inst.scale.x, targetScale, 3, delta))
  })

  // Build lattice matrices once.
  const matrices = useMemo(() => {
    const list: THREE.Matrix4[] = []
    const spacing = 1.05
    const offset = ((LATTICE - 1) * spacing) / 2
    for (let x = 0; x < LATTICE; x++) {
      for (let y = 0; y < LATTICE; y++) {
        for (let z = 0; z < LATTICE; z++) {
          dummy.position.set(
            (x - 0) * spacing - offset,
            (y - 0) * spacing - offset,
            (z - 0) * spacing - offset,
          )
          const s = 0.28 + ((x + y + z) % 3) * 0.06
          dummy.scale.setScalar(s)
          dummy.updateMatrix()
          list.push(dummy.matrix.clone())
        }
      }
    }
    return list
  }, [dummy])

  const setup = (inst: THREE.InstancedMesh | null) => {
    if (!inst) return
    instRef.current = inst
    matrices.forEach((m, i) => inst.setMatrixAt(i, m))
    // Pseudo-random emissive intensity per cell.
    matrices.forEach((_, i) => {
      const t = (i * 2654435761) % 1000 / 1000
      color.lerpColors(new THREE.Color(PRIMARY_DIM), new THREE.Color(PRIMARY), 0.15 + t * 0.85)
      inst.setColorAt(i, color)
    })
    inst.instanceMatrix.needsUpdate = true
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true
  }

  return (
    <Float speed={1.4} rotationIntensity={0.25} floatIntensity={0.6}>
      <instancedMesh ref={setup} args={[undefined, undefined, count]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          emissive={PRIMARY}
          emissiveIntensity={0.9}
          roughness={0.35}
          metalness={0.1}
          transparent
          opacity={0.92}
        />
      </instancedMesh>
    </Float>
  )
}

/** Outer wireframe shell that counter-rotates against the lattice. */
function WireShell() {
  const ref = useRef<THREE.LineSegments>(null!)
  const geometry = useMemo(() => {
    const box = new THREE.BoxGeometry(4.6, 4.6, 4.6)
    return new THREE.EdgesGeometry(box)
  }, [])

  useFrame((state, delta) => {
    ref.current.rotation.y -= delta * 0.08
    ref.current.rotation.x = THREE.MathUtils.damp(
      ref.current.rotation.x,
      state.pointer.y * 0.3,
      2,
      delta,
    )
  })

  return (
    <lineSegments ref={ref} geometry={geometry}>
      <lineBasicMaterial color={PRIMARY} transparent opacity={0.28} />
    </lineSegments>
  )
}

/** Ambient GPU particle field. */
function Particles({ count = 220 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null!)

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const r = 3.5 + Math.random() * 3.5
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      arr[i * 3 + 2] = r * Math.cos(phi)
    }
    return arr
  }, [count])

  useFrame((state, delta) => {
    ref.current.rotation.y += delta * 0.03
    ref.current.rotation.x = state.pointer.y * 0.1
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.035} color={ACCENT} transparent opacity={0.6} sizeAttenuation />
    </points>
  )
}

/**
 * Mouse-reactive WebGL scene: a glowing matrix lattice inside a wireframe
 * shell, orbited by GPU particles. Rendered on the landing hero.
 */
export function Hero3DCanvas({ className }: { className?: string }) {
  return (
    <div className={cn('relative h-full w-full', className)} aria-hidden>
      {/* Fallback glow behind the canvas */}
      <div className="absolute inset-0 rounded-full bg-primary/10 blur-[100px]" />
      <Canvas
        camera={{ position: [0, 0, 9], fov: 42 }}
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        className="!absolute inset-0"
      >
        <ambientLight intensity={0.4} />
        <pointLight position={[6, 6, 8]} intensity={30} color={PRIMARY} />
        <pointLight position={[-6, -4, 4]} intensity={18} color={ACCENT} />
        <MatrixLattice />
        <WireShell />
        <Particles />
      </Canvas>
    </div>
  )
}
