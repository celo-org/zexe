pub mod error {
    pub type Result<T> = T;
}

pub struct Context {}

pub type DeviceMemory<T> = Vec<T>;
