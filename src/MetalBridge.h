#pragma once
// ═══════════════════════════════════════════════════════════════
// MetalBridge.h — C interface for Obj-C Metal/SDL bridge
//
// This is the ONLY file that touches both Cocoa and Metal ObjC.
// It exposes a pure C interface so the rest of the project
// can stay in pure C++ with metal-cpp.
// ═══════════════════════════════════════════════════════════════

#ifdef __cplusplus
extern "C" {
#endif

struct SDL_Window;

// Attaches a CAMetalLayer to the SDL window's NSView.
// Returns the CAMetalLayer* as a void* (caller casts to CA::MetalLayer*).
// The MTLDevice* is also passed as void*.
void* MetalBridge_AttachLayer(struct SDL_Window* window, void* mtlDevice);

#ifdef __cplusplus
}
#endif
