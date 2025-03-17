actor User {}

allow(user: User, _action, _resource) if user.role = "admin";

allow(user, "get_all_users", _resource) if user.role = "admin";

allow(user, "get_all_users_with_role", _resource) if user.role = "admin";

allow(user, "update_role", _resource) if user.role = "admin";

allow(user: User, "view_profile", resource) if user.id = resource.id and user.role = "user";

# Allow admins to edit any profile
allow(user: User, "edit_profile", _resource) if user.role = "admin";

# Allow users to edit only their own profile
allow(user: User, "edit_profile", resource) if user.email = resource.email and user.role = "user";

allow(user: User, "delete_user", resource) if user.id = resource.id and user.role = "user";
