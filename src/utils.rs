use rand::Rng;
use std::fs::File;
use std::io::Write;
use std::path::Path;

//generates a random float between 0 and 1
// pub fn generate_random_number() -> f32 {
//     let mut rng = rand::thread_rng();
//     rng.gen()
// }

//later, this will take a buffer of pixels and write them to a file
pub fn write_to_ppm(filename: &str) {
    const IMAGE_WIDTH: i32 = 256;
    const IMAGE_HEIGHT: i32 = 256;

    //create file + output directory, write metadata into file
    let path = Path::new("output").join(filename);
    let mut file = File::create(path).expect("Failed to create file");
    write!(file, "P3\n").expect("Failed to write to file");
    write!(file, "{} {}\n", IMAGE_WIDTH, IMAGE_HEIGHT)
        .expect("Failed to write Width/Height to file");
    write!(file, "255\n").expect("Failed to write RGB size to file");

    for j in (0..IMAGE_HEIGHT).rev() {
        for i in 0..IMAGE_WIDTH {
            let r = i as f32 / (IMAGE_WIDTH - 1) as f32;
            let g = j as f32 / (IMAGE_HEIGHT - 1) as f32;
            let b = 0.25;

            let ir = (255.999 * r) as i32;
            let ig = (255.999 * g) as i32;
            let ib = (255.999 * b) as i32;

            //write pixel vals into file
            write!(file, "{} {} {}\n", ir, ig, ib).expect("Failed to write pixel values to file");
        }
    }
}
