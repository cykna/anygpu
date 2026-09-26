/// How one struct member reads from, and writes into, the struct's own buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// A single element, e.g. `get intensity(): number { return this.buffer[3]; }`.
    Element { index: u32 },
    /// A nested view class over `[start, end)`, e.g. `get position(): Vector4f32`.
    View { start: u32, end: u32 },
    /// An array view, e.g. `get items(): LightArray { ... }`.
    Array {
        start: u32,
        stride: u32,
        count: Option<usize>,
    },
}

impl Access {
    /// The expression a member's getter returns, given the member's TS type.
    pub fn getter(&self, ty: &str) -> String {
        match *self {
            Access::Element { index } => format!("this.buffer[{index}]"),
            Access::View { start, end } => {
                format!("{ty}.view(this.buffer.subarray({start}, {end}))")
            }
            // A runtime-sized array runs to the end of the struct's buffer, so it
            // is left unbounded; a fixed one is clipped to its own count.
            Access::Array {
                start,
                stride,
                count,
            } => {
                let window = match count {
                    Some(count) => format!("{start}, {}", start + stride * count as u32),
                    None => format!("{start}"),
                };
                format!("new {ty}(this.buffer.subarray({window}), {stride})")
            }
        }
    }

    /// The constructor statement that copies `param` into the struct's buffer.
    pub fn copy(&self, param: &str, offset: u32, width: u32) -> String {
        match *self {
            Access::Element { index } => format!("this.buffer[{index}] = {param};"),
            Access::View { start, .. } | Access::Array { start, .. } => format!(
                "this.buffer.set({param}.buffer, {start}); // offset {offset} bytes / {width} = {start}"
            ),
        }
    }
}

/// A `get`/`set` pair over one element of `this.buffer`.
#[derive(Debug, Clone)]
pub struct Accessor {
    pub name: String,
    pub ty: String,
    /// The element index, already converted from the schema's byte offset.
    pub index: u32,
}

/// A read-only property, e.g. `get length(): number { return ...; }`.
#[derive(Debug, Clone)]
pub struct Getter {
    pub name: String,
    pub ty: String,
    /// The expression after `return`.
    pub expr: String,
}

/// A method, with an optional return type.
#[derive(Debug, Clone)]
pub struct Method {
    pub name: String,
    pub ty: Option<String>,
    pub params: String,
    pub body: Vec<String>,
}

/// The declaration of one generated class.
///
/// Every class shape in the schema — vector, matrix, array, struct — is
/// described this way and rendered by the single [`emit_class`] writer, so the
/// class shell (field, constructor, view factory) is written exactly once.
#[derive(Debug, Default, Clone)]
pub struct ClassDescriptor {
    pub name: String,
    /// The backing array type. It alone decides the `buffer` field, the view
    /// factory and the constructor's buffer parameter.
    pub buffer: String,
    /// Extra fields declared right after `buffer`.
    pub fields: Vec<String>,

    pub params: Vec<String>,
    /// Constructor statements, in the order they run.
    pub init: Vec<String>,
    pub accessors: Vec<Accessor>,
    pub methods: Vec<Method>,
    pub getters: Vec<Getter>,
}

impl ClassDescriptor {
    /// A class that wraps a buffer it does not own: `new X(buffer)`.
    pub fn adopting(name: String, buffer: &str) -> Self {
        Self {
            name,
            buffer: buffer.to_string(),
            params: vec![format!("buffer: {buffer}")],
            init: vec!["this.buffer = buffer;".to_string()],
            ..Default::default()
        }
    }
}
