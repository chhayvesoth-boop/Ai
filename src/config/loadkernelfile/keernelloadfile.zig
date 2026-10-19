const std = @import("std");
const windows = std.os.windows;

//extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) callconv(.winapi) ?windows.HMODULE;
extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) callconv(.winapi) ?windows.HMODULE;
extern "kernel32" fn FreeLibrary(hLibModule: windows.HMODULE) callconv(.winapi) windows.BOOL;
extern "kernel32" fn GetProcAddress(hModule: windows.HMODULE, lpProcName: [*:0]const u8) callconv(.winapi) ?*anyopaque;

// CUDA Type Aliases
pub const CUcontext = ?*anyopaque;
pub const CUdevice = c_int;
pub const CUdeviceptr = u64;
pub const CUmodule = ?*anyopaque;
pub const CUfunction = ?*anyopaque;
pub const CUresult = enum(c_uint) {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    _,
};

// Function Pointer Signatures

//*******************************************************
//*                  chack gpu                          *
//*******************************************************

pub const CuInitFn = *const fn (flags: c_uint) callconv(.c) CUresult;
pub const CuDeviceGetFn = *const fn (device: *CUdevice, ordinal: c_int) callconv(.c) CUresult;
pub const CuDeviceTotalMemFn = *const fn (bytes: *usize, dev: CUdevice) callconv(.c) CUresult;

//*******************************************************
//*               gpuprocessor                          *
//*******************************************************

pub const CuLaunchKernelFn = *const fn (f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, hStream: ?*anyopaque, kernelParams: ?*?*anyopaque, extra: ?*?*anyopaque) callconv(.c) CUresult;
pub const cuModuleLoadfn = *const fn (module: *CUmodule, fname: [*:0]const u8) callconv(.c) CUresult;
pub const cuModuleLoadDatafn = *const fn (module: *CUmodule, image: *const anyopaque) callconv(.c) CUresult;
pub const cuModuleGetFunctionfn = *const fn (hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) callconv(.c) CUresult;
pub const cuModuleUnloadfn = *const fn (hmod: CUmodule) callconv(.c) CUresult;

//*******************************************************
//*               memory menagement                     *
//*******************************************************

pub const cuMemAllocfn = *const fn (dptr: *CUdeviceptr, bytesize: usize) callconv(.c) CUresult;
pub const cuMemFreefn = *const fn (dptr: CUdeviceptr) callconv(.c) CUresult;
pub const cuCtxCreatefn = *const fn (ctx: *CUcontext, flags: c_uint, dev: CUdevice) callconv(.c) CUresult;
pub const cuCtxDestroyfn = *const fn (ctx: CUcontext) callconv(.c) CUresult;
pub const cuMemcpyHtoDfn = *const fn (dstDevice: CUdeviceptr, srcHost: *anyopaque, ByteCount: usize) callconv(.c) CUresult; //cpu to gpu
pub const cuMemcpyDtoHfn = *const fn (dstHost: *anyopaque, srcDevice: CUdeviceptr, ByteCount: usize) callconv(.c) CUresult; //gpu to cpu

pub const checkaviable = struct {
    handle: windows.HMODULE,
    cuInit: CuInitFn,
    cuDeviceGet: CuDeviceGetFn,
    cuDeviceTotalMem: CuDeviceTotalMemFn,
    pub fn load() !checkaviable {
        const handle = LoadLibraryA("nvcuda.dll") orelse {
            std.debug.print("Failed to load nvcuda.dll\n", .{});
            return error.LibraryNotFound;
        };
        errdefer _ = FreeLibrary(handle);

        return .{
            .handle = handle,
            .cuInit = @ptrCast(try resolve(handle, "cuInit")), //
            .cuDeviceGet = @ptrCast(try resolve(handle, "cuDeviceGet")),
            .cuDeviceTotalMem = @ptrCast(try resolve(handle, "cuDeviceTotalMem_v2")), // Note: modern 64-bit CUDA exports the _v2 symbol
        };
    }

    pub fn deinit(self: *checkaviable) void {
        _ = FreeLibrary(self.handle);
    }
};
pub const processergpu = struct {
    handle: windows.HMODULE,
    cuLaunchKernel: CuLaunchKernelFn,
    cuModuleLoad: cuModuleLoadfn,
    cuModuleLoadData: cuModuleLoadDatafn,
    cuModuleGetFunction: cuModuleGetFunctionfn,
    cuModuleUnload: cuModuleUnloadfn,
    pub fn load() !processergpu {
        const handle = LoadLibraryA("nvcuda.dll") orelse {
            std.debug.print("Failed to load nvcuda.dll\n", .{});
            return error.LibraryNotFound;
        };
        errdefer _ = FreeLibrary(handle);

        return .{
            .handle = handle,
            .cuLaunchKernel = @ptrCast(try resolve(handle, "cuLaunchKernel")),
            .cuModuleLoad = @ptrCast(try resolve(handle, "cuModuleLoad")),
            .cuModuleLoadData = @ptrCast(try resolve(handle, "cuModuleLoadData")),
            .cuModuleGetFunction = @ptrCast(try resolve(handle, "cuModuleGetFunction")),
            .cuModuleUnload = @ptrCast(try resolve(handle, "cuModuleUnload")),
        };
    }

    pub fn deinit(self: *processergpu) void {
        _ = FreeLibrary(self.handle);
    }
};

pub const memorymenagement = struct {
    handle: windows.HMODULE,
    cuMemAlloc: cuMemAllocfn,
    cuMemFree: cuMemFreefn,
    cuCtxCreate: cuCtxCreatefn,
    cuCtxDestroy: cuCtxDestroyfn,
    cuMemcpyHtoD: cuMemcpyHtoDfn,
    cuMemcpyDtoH: cuMemcpyDtoHfn,
    pub fn load() !memorymenagement {
        const handle = LoadLibraryA("nvcuda.dll") orelse {
            std.debug.print("Failed to load nvcuda.dll\n", .{});
            return error.LibraryNotFound;
        };
        errdefer _ = FreeLibrary(handle);

        return .{
            .handle = handle,
            .cuMemAlloc = @ptrCast(try resolve(handle, "cuMemAlloc_v2")),
            .cuMemFree = @ptrCast(try resolve(handle, "cuMemFree_v2")),
            .cuCtxCreate = @ptrCast(try resolve(handle, "cuCtxCreate_v2")),
            .cuCtxDestroy = @ptrCast(try resolve(handle, "cuCtxDestroy_v2")),
            .cuMemcpyHtoD = @ptrCast(try resolve(handle, "cuMemcpyHtoD_v2")),
            .cuMemcpyDtoH = @ptrCast(try resolve(handle, "cuMemcpyDtoH_v2")),
        };
    }

    pub fn deinit(self: *memorymenagement) void {
        _ = FreeLibrary(self.handle);
    }
};

//************************************************************************
//resolve function to get the address of a symbol from the loaded library*
//************************************************************************
fn resolve(handle: windows.HMODULE, name: [*:0]const u8) !*anyopaque {
    return GetProcAddress(handle, name) orelse {
        std.debug.print("Failed to resolve symbol: {s}\n", .{name});
        return error.SymbolNotFound;
    };
}
