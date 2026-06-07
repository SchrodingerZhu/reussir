#ifndef REUSSIR_SYNTAX_H
#define REUSSIR_SYNTAX_H

#ifdef __cplusplus
extern "C" {
#endif

// Parse complete Reussir source and return a JSON response.
// Success shape: {"ok":true,"value":...}
// Error shape: {"ok":false,"diagnostic":"...ariadne report..."}
// The returned string is owned by the syntax library and must be released with
// reussir_syntax_string_free.
char *reussir_syntax_parse_program_json(const char *input, const char *file_name);
char *reussir_syntax_parse_stmt_json(const char *input, const char *file_name);
char *reussir_syntax_parse_expr_json(const char *input, const char *file_name);
char *reussir_syntax_parse_type_json(const char *input, const char *file_name);
void reussir_syntax_string_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_SYNTAX_H
