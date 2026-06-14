//! Miscellaneous and foreign-function-interface constructors: `reussir.scf.yield`,
//! `reussir.panic`, `reussir.trampoline`, and `reussir.polyffi`.

use crate::context::{Attr, Context};
use crate::ir::{Op, Value};

impl<'a> Context<'a> {
    /// `reussir.scf.yield` — yield from a `record.dispatch`/`nullable.dispatch`
    /// region.
    pub fn reussir_scf_yield(&self, values: &[Value<'a>]) -> Op<'a> {
        self.op("reussir.scf.yield").operands(values).build_zero()
    }

    /// `reussir.panic` — abort with a message.
    pub fn panic(&self, message: &str) -> Op<'a> {
        let attr = self.attr_str(message);
        self.op("reussir.panic")
            .attrs(self.attr_dict(&[("message", attr)]))
            .build_zero()
    }

    /// `reussir.trampoline` — declare an exported (`direction = 1`) or imported
    /// (`direction = 0`) C-ABI trampoline named `sym_name` for `target` under the
    /// given ABI name.
    pub fn trampoline(
        &self,
        sym_name: &str,
        target: &str,
        abi_name: &str,
        direction: i32,
    ) -> Op<'a> {
        let i32 = self.int(32);
        let attrs = self.attr_dict(&[
            ("abi_name", self.attr_str(abi_name)),
            ("direction", self.attr_int(direction as i64, i32)),
            ("sym_name", self.attr_str(sym_name)),
            ("target", self.attr_symbol(target)),
        ]);
        self.op("reussir.trampoline").attrs(attrs).build_zero()
    }

    /// `reussir.polyffi` — a polymorphic FFI stub carrying a module texture
    /// template and a substitution dictionary.
    pub fn polyffi(&self, module_texture: &str, substitutions: Attr<'a>) -> Op<'a> {
        let attrs = self.attr_dict(&[
            ("moduleTexture", self.attr_str(module_texture)),
            ("substitutions", substitutions),
        ]);
        self.op("reussir.polyffi").attrs(attrs).build_zero()
    }
}
