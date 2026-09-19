'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame } from '@react-three/fiber'
import { Float } from '@react-three/drei'
import { cn } from '@/lib/utils'

const PRIMARY = '#4ade80'
const PRIMARY_DIM = '#065f46'
const ACCENT = '#22d3ee'
const ACCENT_BRIGHT = '#67e8f9'
const LIME = '#a3e635'

const LATTICE = 4 // 4x4x4 tensor lattice = 64 cells

/**
 * 4x4x4 Holographic Tensor Lattice:
 * - Cells pulsate in mathematical waves (representing tensor activations)
 * - Mouse cursor repels and excites nearby nodes
 * - Window scroll smoothly dissects / explodes the tensor layers
 */
function HolographicTensorCore({ scrollProgress }: { scrollProgress: number }) {
  const instRef = useRef<THREE.InstancedMesh>(null!)
  const count = LATTICE ** 3

  const dummy = useMemo(() => new THREE.Object3D(), [])
  const cellColor = useMemo(() => new THREE.Color(), [])
  const targetColor = useMemo(() => new THREE.Color(), [])

  // Base positions
  const basePositions = useMemo(() => {
    const list: { pos: THREE.Vector3; baseScale: number; phase: number }[] = []
    const spacing = 1.05
    const offset = ((LATTICE - 1) * spacing) / 2
    for (let x = 0; x < LATTICE; x++) {
      for (let y = 0; y < LATTICE; y++) {
        for (let z = 0; z < LATTICE; z++) {
          const pos = new THREE.Vector3(
            x * spacing - offset,
            y * spacing - offset,
            z * spacing - offset,
          )
          const baseScale = 0.28 + ((x + y + z) % 3) * 0.05
          const phase = (x * 3 + y * 5 + z * 7) * 0.4
          list.push({ pos, baseScale, phase })
        }
      }
    }
    return list
  }, [])

  useFrame((state, delta) => {
    const inst = instRef.current
    if (!inst) return

    const time = state.clock.getElapsedTime()

    // Base auto-rotation + mouse damping + scroll spin acceleration
    const scrollSpin = scrollProgress * 1.5
    inst.rotation.y += delta * (0.2 + scrollSpin * 0.5)
    inst.rotation.x = THREE.MathUtils.damp(
      inst.rotation.x,
      -state.pointer.y * 0.65 + scrollProgress * 0.4,
      2.5,
      delta,
    )
    inst.rotation.z = THREE.MathUtils.damp(
      inst.rotation.z,
      state.pointer.x * 0.35,
      2.5,
      delta,
    )

    // Tensor expansion based on scroll (layers separate to reveal inner core)
    const expansion = 1 + scrollProgress * 1.25

    // Update each node in the 4x4x4 tensor
    basePositions.forEach((node, i) => {
      // Wave breathing
      const wave = Math.sin(time * 3 + node.phase) * 0.08
      const s = node.baseScale + wave

      // Expanded position
      dummy.position.copy(node.pos).multiplyScalar(expansion)

      // Slight mouse repulsion in normalized space
      const pointerDist = Math.hypot(
        state.pointer.x * 3 - dummy.position.x * 0.4,
        state.pointer.y * 3 - dummy.position.y * 0.4,
      )
      if (pointerDist < 1.8) {
        const factor = (1.8 - pointerDist) * 0.25
        dummy.position.x += state.pointer.x * factor
        dummy.position.y += state.pointer.y * factor
        dummy.scale.setScalar(s * (1 + factor * 0.8))
      } else {
        dummy.scale.setScalar(s)
      }

      dummy.updateMatrix()
      inst.setMatrixAt(i, dummy.matrix)

      // Color animation: ripple of cyber-emerald and neon-cyan
      const t = (Math.sin(time * 2 + node.phase) + 1) * 0.5
      targetColor.lerpColors(new THREE.Color(PRIMARY), new THREE.Color(ACCENT), t)
      if (pointerDist < 1.8) {
        targetColor.lerp(new THREE.Color(LIME), (1.8 - pointerDist) * 0.6)
      }
      inst.setColorAt(i, targetColor)
    })

    inst.instanceMatrix.needsUpdate = true
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true
  })

  return (
    <Float speed={1.8} rotationIntensity={0.25} floatIntensity={0.5}>
      <instancedMesh ref={instRef} args={[undefined, undefined, count]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          emissive={PRIMARY}
          emissiveIntensity={1.2}
          roughness={0.2}
          metalness={0.8}
          transparent
          opacity={0.88}
        />
      </instancedMesh>
    </Float>
  )
}

/**
 * Concentric CUDA Execution Gyroscope Rings:
 * - 3 outer rotating cyber rings representing CUDA Streams
 * - Scroll opens up the gyroscope angle
 */
function StreamGyroscopeRings({ scrollProgress }: { scrollProgress: number }) {
  const ring1 = useRef<THREE.Group>(null!)
  const ring2 = useRef<THREE.Group>(null!)
  const ring3 = useRef<THREE.Group>(null!)

  const ringGeo1 = useMemo(() => new THREE.TorusGeometry(3.6, 0.02, 16, 100), [])
  const ringGeo2 = useMemo(() => new THREE.TorusGeometry(4.2, 0.025, 16, 100), [])
  const ringGeo3 = useMemo(() => new THREE.TorusGeometry(4.8, 0.02, 16, 100), [])

  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime()
    const mouseX = state.pointer.x * 0.3
    const mouseY = state.pointer.y * 0.3

    if (ring1.current) {
      ring1.current.rotation.x = time * 0.35 + mouseY
      ring1.current.rotation.y = time * 0.2 + mouseX
      const scale = 1 + scrollProgress * 0.3
      ring1.current.scale.setScalar(scale)
    }
    if (ring2.current) {
      ring2.current.rotation.x = -time * 0.25 - mouseY
      ring2.current.rotation.z = time * 0.4 + mouseX
      const scale = 1 + scrollProgress * 0.4
      ring2.current.scale.setScalar(scale)
    }
    if (ring3.current) {
      ring3.current.rotation.y = -time * 0.3 + mouseX
      ring3.current.rotation.z = -time * 0.15 + mouseY
      const scale = 1 + scrollProgress * 0.5
      ring3.current.scale.setScalar(scale)
    }
  })

  return (
    <>
      <group ref={ring1}>
        <mesh geometry={ringGeo1}>
          <meshBasicMaterial color={PRIMARY} transparent opacity={0.45} />
        </mesh>
      </group>
      <group ref={ring2}>
        <mesh geometry={ringGeo2}>
          <meshBasicMaterial color={ACCENT} transparent opacity={0.35} />
        </mesh>
      </group>
      <group ref={ring3}>
        <mesh geometry={ringGeo3}>
          <meshBasicMaterial color={LIME} transparent opacity={0.25} />
        </mesh>
      </group>
    </>
  )
}

/**
 * Geometric Wireframe Bounding Shell (Tesseract / Hypercube cage).
 */
function WireHypercube({ scrollProgress }: { scrollProgress: number }) {
  const ref = useRef<THREE.LineSegments>(null!)
  const geometry = useMemo(() => {
    const box = new THREE.BoxGeometry(4.8, 4.8, 4.8)
    return new THREE.EdgesGeometry(box)
  }, [])

  useFrame((state, delta) => {
    if (!ref.current) return
    ref.current.rotation.y -= delta * 0.08
    ref.current.rotation.x = THREE.MathUtils.damp(
      ref.current.rotation.x,
      state.pointer.y * 0.35,
      2,
      delta,
    )
    const scale = 1 + scrollProgress * 0.4
    ref.current.scale.setScalar(scale)
  })

  return (
    <lineSegments ref={ref} geometry={geometry}>
      <lineBasicMaterial color={PRIMARY} transparent opacity={0.3} />
    </lineSegments>
  )
}

/**
 * Pulsing Synapse Compute Lines (GEMM / Autograd data pulses).
 */
function SynapseDataBeams() {
  const linesRef = useRef<THREE.LineSegments>(null!)

  const geometry = useMemo(() => {
    const coords: number[] = []
    const spacing = 1.05
    const offset = ((LATTICE - 1) * spacing) / 2
    // Connect diagonal cross nodes
    for (let i = 0; i < 18; i++) {
      const x1 = (Math.floor(Math.random() * 4) * spacing) - offset
      const y1 = (Math.floor(Math.random() * 4) * spacing) - offset
      const z1 = (Math.floor(Math.random() * 4) * spacing) - offset

      const x2 = (Math.floor(Math.random() * 4) * spacing) - offset
      const y2 = (Math.floor(Math.random() * 4) * spacing) - offset
      const z2 = (Math.floor(Math.random() * 4) * spacing) - offset

      coords.push(x1, y1, z1, x2, y2, z2)
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(coords, 3))
    return geo
  }, [])

  useFrame((state) => {
    if (!linesRef.current) return
    const time = state.clock.getElapsedTime()
    // Flashing synapse intensity
    const mat = linesRef.current.material as THREE.LineBasicMaterial
    mat.opacity = 0.15 + Math.sin(time * 6) * 0.12
  })

  return (
    <lineSegments ref={linesRef} geometry={geometry}>
      <lineBasicMaterial color={ACCENT_BRIGHT} transparent opacity={0.2} />
    </lineSegments>
  )
}

/**
 * High-Density Quantum Compute Particles (650 particles with vortex swirl).
 */
function QuantumComputeParticles({ count = 650 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null!)

  const [positions, initialRadii] = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const radii = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      const r = 3.0 + Math.random() * 4.5
      radii[i] = r
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i * 3 + 2] = r * Math.cos(phi)
    }
    return [pos, radii]
  }, [count])

  useFrame((state, delta) => {
    if (!ref.current) return
    // Swirl with mouse velocity
    ref.current.rotation.y += delta * 0.05 + state.pointer.x * 0.002
    ref.current.rotation.x = THREE.MathUtils.damp(
      ref.current.rotation.x,
      state.pointer.y * 0.25,
      2,
      delta,
    )
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.045}
        color={ACCENT}
        transparent
        opacity={0.65}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

/**
 * Dynamic Mouse-Follow Point Light (casts realistic 3D specular glints).
 */
function CursorTrackingLight() {
  const lightRef = useRef<THREE.PointLight>(null!)

  useFrame((state, delta) => {
    if (!lightRef.current) return
    const targetX = state.pointer.x * 6
    const targetY = state.pointer.y * 6
    lightRef.current.position.x = THREE.MathUtils.damp(
      lightRef.current.position.x,
      targetX,
      3,
      delta,
    )
    lightRef.current.position.y = THREE.MathUtils.damp(
      lightRef.current.position.y,
      targetY,
      3,
      delta,
    )
  })

  return <pointLight ref={lightRef} position={[0, 0, 7]} intensity={35} color={PRIMARY} />
}

/**
 * Master Mouse & Scroll Reactive WebGL Scene.
 */
export function Hero3DCanvas({ className }: { className?: string }) {
  const [scrollProgress, setScrollProgress] = useState(0)

  // Track window scroll progress for 3D dissection morph
  useEffect(() => {
    let ticking = false
    const onScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const maxScroll = 600
          const current = Math.min(1, Math.max(0, window.scrollY / maxScroll))
          setScrollProgress(current)
          ticking = false
        })
        ticking = true
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className={cn('relative h-full w-full', className)} aria-hidden>
      {/* Background Holographic Aurora Glow */}
      <div className="absolute inset-0 rounded-full bg-primary/15 blur-[120px] animate-pulse-glow" />
      <div className="absolute inset-x-10 bottom-0 h-40 bg-accent/10 blur-[90px]" />

      <Canvas
        camera={{ position: [0, 0, 9.2], fov: 42 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        className="!absolute inset-0"
      >
        <ambientLight intensity={0.4} />
        <CursorTrackingLight />
        <pointLight position={[-7, -5, 5]} intensity={25} color={ACCENT} />
        <pointLight position={[7, -5, -4]} intensity={18} color={LIME} />

        <HolographicTensorCore scrollProgress={scrollProgress} />
        <StreamGyroscopeRings scrollProgress={scrollProgress} />
        <WireHypercube scrollProgress={scrollProgress} />
        <SynapseDataBeams />
        <QuantumComputeParticles />
      </Canvas>
    </div>
  )
}
