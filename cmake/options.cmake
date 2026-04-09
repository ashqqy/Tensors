set(COMPILE_OPT
  $<$<CONFIG:Debug>:-DDEBUG -g -O0 -pedantic -Wall -Werror -Wextra -fsanitize=address -fsanitize=undefined>
)

set(LINK_OPT
  $<$<CONFIG:Debug>:-DDEBUG -g -O0 -pedantic -Wall -Werror -Wextra -fsanitize=address -fsanitize=undefined>
)
