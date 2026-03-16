// ═══════════════════════════════════════════════════════════════
// MetalBridge.mm — Obj-C bridge for Metal layer attachment
//
// This file does NOT include metal-cpp headers.
// It uses native Obj-C APIs only.
// ═══════════════════════════════════════════════════════════════

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <SDL.h>
#include <SDL_syswm.h>

#include "MetalBridge.h"

extern "C" void* MetalBridge_AttachLayer(SDL_Window* window, void* mtlDevice) {
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    if (!SDL_GetWindowWMInfo(window, &wmInfo)) {
        return nullptr;
    }

    NSWindow* nsWin = (__bridge NSWindow*)wmInfo.info.cocoa.window;
    NSView* view = [nsWin contentView];

    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = (__bridge id<MTLDevice>)mtlDevice;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;           // We only render to it, don't read back
    layer.displaySyncEnabled = YES;        // VSync — prevents tearing
    layer.maximumDrawableCount = 3;        // Triple buffering

    // Make the layer fill the view properly
    layer.contentsGravity = kCAGravityResize;
    layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;

    [view setWantsLayer:YES];
    [view setLayer:layer];

    return (__bridge void*)layer;
}